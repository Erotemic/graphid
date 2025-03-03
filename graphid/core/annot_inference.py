import copy
import logging
import collections
import ubelt as ub
import numpy as np
import pandas as pd
import networkx as nx
import itertools as it
from graphid import util
from graphid.core import state as const
from graphid.core import mixin_viz
from graphid.core import mixin_helpers
from graphid.core import mixin_dynamic
from graphid.core import mixin_priority
from graphid.core import mixin_loops
from graphid.core import mixin_callbacks
from graphid.core import mixin_simulation
from graphid.core import mixin_invariants
from graphid.core import mixin_redundancy
from graphid.core.state import POSTV, NEGTV, INCMP, UNREV, UNKWN
from graphid.core.state import UNINFERABLE
from graphid.core.state import SAME, DIFF, NULL


def _rectify_decision(evidence_decision, meta_decision):
    """
    If evidence decision is not explicitly set, then meta decision is used to
    make a guess. Raises a ValueError if decisions are in incompatible states.
    """
    # Default to the decision based on the media evidence
    decision = evidence_decision
    # Overwrite the graph decision with the meta decision if necessary
    if meta_decision == SAME:
        if decision in UNINFERABLE:
            decision = POSTV
        elif decision == NEGTV:
            raise ValueError('evidence=negative and meta=same')
    elif meta_decision == DIFF:
        if decision in UNINFERABLE:
            decision = NEGTV
        elif decision == POSTV:
            raise ValueError('evidence=positive and meta=diff')
    return decision


class Consistency:
    def is_consistent(infr, cc):
        """
        Determines if a PCC contains inconsistencies

        Args:
            cc (set): nodes in a PCC

        Returns:
            flag: bool: returns True unless cc contains any negative edges

        Example:
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(num_pccs=1, p_incon=1)
            >>> assert not infr.is_consistent(next(infr.positive_components()))
            >>> infr = demo.demodata_infr(num_pccs=1, p_incon=0)
            >>> assert infr.is_consistent(next(infr.positive_components()))
        """
        return len(cc) <= 2 or not any(util.edges_inside(infr.neg_graph, cc))

    def positive_components(infr, graph=None):
        """
        Generates the positive connected compoments (PCCs) in the graph
        These will contain both consistent and inconsinstent PCCs.

        Yields:
            cc: set: nodes within the PCC
        """
        pos_graph = infr.pos_graph
        if graph is None or graph is infr.graph:
            ccs = pos_graph.connected_components()
        else:
            unique_labels = {
                pos_graph.node_label(node) for node in graph.nodes()}
            ccs = (pos_graph.connected_to(node) for node in unique_labels)
        for cc in ccs:
            yield cc

    def inconsistent_components(infr, graph=None):
        """
        Generates inconsistent PCCs.
        These PCCs contain internal negative edges indicating an error exists.
        """
        for cc in infr.positive_components(graph):
            if not infr.is_consistent(cc):
                yield cc

    def consistent_components(infr, graph=None):
        """
        Generates consistent PCCs.
        These PCCs contain no internal negative edges.

        Yields:
            cc: set: nodes within the PCC
        """
        # Find PCCs without any negative edges
        for cc in infr.positive_components(graph):
            if infr.is_consistent(cc):
                yield cc


class Feedback:
    def _check_edge(infr, edge):
        aid1, aid2 = edge
        if aid1 not in infr.aids_set:
            raise ValueError('aid1=%r is not part of the graph' % (aid1,))
        if aid2 not in infr.aids_set:
            raise ValueError('aid2=%r is not part of the graph' % (aid2,))

    def add_feedback_from(infr, items, verbose=None, **kwargs):
        """
        Args:
            items (List[Edge]): each edge is a dictionary with
                aid1, aid2, evidence_decision, meta_decision, etc..
        """
        if verbose is None:
            verbose = infr.verbose > 5
        if isinstance(items, pd.DataFrame):
            if list(items.index.names) == ['aid1', 'aid2']:
                for edge, data in items.iterrows():
                    infr.add_feedback(edge=edge, verbose=verbose, **data)
            else:
                raise ValueError(
                    'Cannot interpret pd.DataFrame without edge index')
        else:
            # Dangerous if item length > 3
            for item in items:
                args = []
                if len(item) == 1:
                    # Case where items=[edge1, edge2]
                    if isinstance(item[0], int) or len(item[0]) != 2:
                        raise ValueError('invalid edge')
                if len(item) == 2:
                    # Case where items=[(edge1, state), (edge2, state)]
                    if ub.iterable(item[0]):
                        edge = item[0]
                        args = item[1:]
                    else:
                        edge = item
                else:
                    raise ValueError('invalid edge')
                    # Case where items=[(u, v, state), (u, v, state)]
                if len(item) > 3:
                    raise ValueError('pass in data as a dataframe or '
                                     'use kwargs')
                infr.add_feedback(edge, *args, verbose=verbose, **kwargs)

    def edge_decision(infr, edge):
        """
        Gets a decision on an edge, either explicitly or implicitly
        """
        evidence_decision = infr.get_edge_attr(edge, 'evidence_decision',
                                               on_missing='default',
                                               default=UNREV)
        meta_decision = infr.get_edge_attr(edge, 'meta_decision',
                                           on_missing='default', default=NULL)
        decision = _rectify_decision(evidence_decision, meta_decision)
        return decision

    def edge_decision_from(infr, edges):
        """
        Gets a decision for multiple edges
        """
        edges = list(edges)
        evidence_decisions = infr.gen_edge_values(
            'evidence_decision', edges, on_missing='default', default=UNREV)
        meta_decisions = infr.gen_edge_values(
            'meta_decision', edges, on_missing='default', default=NULL)
        for ed, md in zip(evidence_decisions, meta_decisions):
            yield _rectify_decision(ed, md)

    def add_node_feedback(infr, aid, **attrs):
        infr.print('Setting aid={} {}'.format(aid, ub.urepr(attrs)))
        for key, value in attrs.items():
            infr.set_node_attrs(key, {aid: value})

    def add_feedback(infr, edge, evidence_decision=None, tags=None,
                     user_id=None, meta_decision=None, confidence=None,
                     timestamp_c1=None, timestamp_c2=None, timestamp_s1=None,
                     timestamp=None, verbose=None, priority=None):
        """
        Primary method for adding feedback and review edges to the graph.

        Args:
            edge (tuple): an undirected edge represented as a pair of aids
            evidence_decision (str): decision made based on visual evidence
                between the two photos. Can be POSTV, NEGTV, INCMP, or UNKWN.
                Note: POSTV etc... are the variables not the strings.
            tags (list of str): additional information to specify
            user_id (str): who is doing this review. This can identify a human
                or algorithm reviewer (e.g. 'user:joncrall' or 'algo:vamp').
            meta_decision (str): decision made based on external knowledge.
                Perhaps the photographer knows that two animals are the same
                because all photos are of the same animal. This constrains
                the identity problem, but does not impact the computer vision
                learning algorithms, which aren't given the info needed to make
                this sort of decision.
            confidence (str): how sure is the user of this decision.
            timestamp_c1 (int): Time that the review client started
            timestamp_c2 (int): Time that the review client ended
            timestamp_s1 (int): Time that the review server started
            timestamp (int): Time that the review server ended
            verbose (bool): verbosity
            priority (float, optional): the priority assigned to this edge
                before review. This is only relevant for the termination
                criterion.

        Notes:
            If `infr.params['inference.enabled']` is True, then the edge is
            inserted into the graph and its properties are updated dynamically.
            Otherwise it is only added to the internal feedback dictionary and
            the `apply_feedback_edges` method must be called.

        Example:
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(num_pccs=5)
            >>> infr.add_feedback((5, 6), POSTV)
            >>> infr.add_feedback((5, 6), NEGTV, tags=['photobomb'])
            >>> infr.add_feedback((1, 2), INCMP)
            >>> print(ub.urepr(infr.internal_feedback, nl=3, sk=1))
            >>> assert len(infr.external_feedback) == 0
            >>> assert len(infr.internal_feedback) == 2
            >>> assert len(infr.internal_feedback[(5, 6)]) == 2
            >>> assert len(infr.internal_feedback[(1, 2)]) == 1
        """
        prev_verbose = infr.verbose
        if verbose is not None:
            infr.verbose = verbose
        edge = aid1, aid2 = util.e_(*edge)

        if not infr.has_edge(edge):
            if infr.params['allow_unseen_nodes']:
                # Allow new aids
                if not infr.graph.has_node(aid1):
                    infr.add_aids([aid1])
                if not infr.graph.has_node(aid2):
                    infr.add_aids([aid2])
            infr._check_edge(edge)
            infr.graph.add_edge(aid1, aid2)

        if evidence_decision is None:
            evidence_decision = UNREV
        if meta_decision is None:
            meta_decision = const.META_DECISION.CODE.NULL
        if confidence is None:
            confidence = const.CONFIDENCE.CODE.UNKNOWN
        if timestamp is None:
            timestamp = ub.timestamp()

        msg = 'add_feedback ({}, {}), '.format(aid1, aid2)
        loc = locals()
        msg += ', '.join([
            str(val)
            for key, val in (
                (key, loc[key])
                for key in ['evidence_decision', 'tags', 'user_id',
                            'confidence', 'meta_decision'])
            if val is not None
        ])
        infr.print(msg, 2, color='white')

        if meta_decision == NULL:
            # TODO: check previous meta_decision and use that if its consistent
            # with the evidence decision.
            pass

        decision = _rectify_decision(evidence_decision, meta_decision)

        if decision == UNREV:
            # Unreviewing an edge deletes anything not yet committed
            if edge in infr.external_feedback:
                raise ValueError('External edge reviews cannot be undone')
            if edge in infr.internal_feedback:
                del infr.internal_feedback[edge]

        # Remove the edge from the queue if it is in there.
        if infr.queue:
            if edge in infr.queue:
                del infr.queue[edge]

        # Keep track of sequential reviews and set properties on global graph
        num_reviews = infr.get_edge_attr(edge, 'num_reviews', default=0)

        review_id = next(infr.review_counter)
        feedback_item = {
            'tags': tags,
            'evidence_decision': evidence_decision,
            'meta_decision': meta_decision,
            'timestamp_c1': timestamp_c1,
            'timestamp_c2': timestamp_c2,
            'timestamp_s1': timestamp_s1,
            'timestamp': timestamp,
            'confidence': confidence,
            'user_id': user_id,
            'num_reviews': num_reviews + 1,
            'review_id': review_id,
        }
        infr.internal_feedback[edge].append(feedback_item)
        infr.set_edge_attr(edge, feedback_item)

        if infr.test_mode:
            prev_decision = infr._get_current_decision(edge)
            infr._dynamic_test_callback(edge, decision, prev_decision, user_id)

        # must happen after dynamic test callback
        infr.set_edge_attr(edge, {'decision': decision})

        if infr.params['inference.enabled']:
            assert infr.dirty is False, (
                'need to recompute before dynamic inference continues')
            # Dynamically update priority queue based on the new edge
            if decision == POSTV:
                action = infr._positive_decision(edge)
            elif decision == NEGTV:
                action = infr._negative_decision(edge)
            elif decision in UNINFERABLE:
                # incomparable and unreview have the same inference structure
                action = infr._uninferable_decision(edge, decision)
            else:
                raise AssertionError('Unknown decision=%r' % (decision,))
            if infr.test_mode:
                infr.test_state['action'] = action
            if False:
                infr._print_debug_ccs()
        else:
            action = None
            infr.dirty = True
            infr._add_review_edge(edge, decision)

        if infr.params['inference.enabled'] and infr.refresh:
            # only add to criteria if this wasn't requested as a fix edge
            if priority is not None and priority <= 1.0:
                meaningful = bool({'merge', 'split'} & set(action))
                infr.refresh.add(meaningful, user_id, decision)

        if infr.test_mode:
            infr.metrics_list.append(infr.measure_metrics())

        infr.verbose = prev_verbose

    def _print_debug_ccs(infr):
        assert all([ub.allsame(infr.node_labels(*cc))
                    for cc in infr.positive_components()])
        sorted_ccs = sorted([
            set(cc) for cc in infr.pos_graph.connected_components()
        ])
        msg = '[' + ', '.join([
            repr(cc)
            if infr.is_consistent(cc) else
            ub.color_text(repr(cc), 'red')
            for cc in sorted_ccs]) + ']'
        print(msg)

    @util.classproperty
    def feedback_keys(Infr):
        """ edge attribute keys used for feedback """
        return Infr.feedback_data_keys + ['num_reviews', 'review_id']

    @util.classproperty
    def feedback_data_keys(Infr):
        """ edge attribute keys used for feedback """
        return [
            'evidence_decision', 'tags', 'user_id',
            'meta_decision', 'timestamp_c1', 'timestamp_c2',
            'timestamp_s1', 'timestamp', 'confidence'
        ]

    def apply_feedback_edges(infr):
        """
        Transforms the feedback dictionaries into nx graph edge attributes.
        This
        """
        infr.print('apply_feedback_edges', 1)
        # Transforms dictionary feedback into numpy array
        edges = []
        attr_lists = {key: [] for key in infr.feedback_keys}
        for edge, vals in infr.all_feedback_items():
            # hack for feedback rectification
            feedback_item = infr._rectify_feedback_item(vals)
            feedback_item['review_id'] = next(infr.review_counter)
            feedback_item['num_reviews'] = len(vals)
            # if feedback_item['decision'] == 'unknown':
            #     continue
            set1 = set(feedback_item.keys())
            set2 = set(attr_lists.keys())
            if set1 != set2:
                raise AssertionError('Bad feedback keys')
            for key, val in feedback_item.items():
                attr_lists[key].append(val)
            edges.append(edge)

        assert ub.allsame(list(map(len, attr_lists.values())))
        assert len(edges) == len(next(iter(attr_lists.values())))

        # Put pair orders in context of the graph
        infr.print('_set_feedback_edges(nEdges=%d)' % (len(edges),), 3)
        # Ensure edges exist
        for edge in edges:
            if not infr.graph.has_edge(*edge):
                infr.graph.add_edge(*edge)

        # take evidence_decision and meta_decision into account
        decisions = [
            _rectify_decision(ed, md) for ed, md in
            zip(attr_lists['evidence_decision'], attr_lists['meta_decision'])
        ]

        for state, es in ub.group_items(edges, decisions).items():
            infr._add_review_edges_from(es, state)

        for key, val_list in attr_lists.items():
            infr.set_edge_attrs(key, ub.dzip(edges, val_list))

        if infr.params['inference.enabled']:
            infr.apply_nondynamic_update()

    def _rectify_feedback(infr, feedback):
        return {edge: infr._rectify_feedback_item(vals)
                for edge, vals in feedback.items()}

    def _rectify_feedback_item(infr, vals):
        """ uses most recently use strategy """
        return vals[-1]

    def all_feedback_items(infr):
        for edge, vals in infr.external_feedback.items():
            yield edge, vals
        for edge, vals in infr.internal_feedback.items():
            yield edge, vals

    def all_feedback(infr):
        all_feedback = ub.ddict(list)
        all_feedback.update(infr.all_feedback_items())
        return all_feedback

    def clear_feedback(infr, edges=None):
        """ Delete all edges properties related to feedback """
        if edges is None:
            edges = infr.graph.edges()
        edges = list(edges)
        infr.print('clear_feedback len(edges) = %r' % (len(edges)), 2)
        infr.external_feedback = ub.ddict(list)
        infr.internal_feedback = ub.ddict(list)

        # Kill all feedback, remote edge labels, but leave graph edges alone
        keys = infr.feedback_keys + ['inferred_state']
        util.nx_delete_edge_attr(infr.graph, keys, edges)

        # Move reviewed edges back into the unreviewed graph
        for key in (POSTV, NEGTV, INCMP):
            subgraph = infr.review_graphs[key]
            prev_edges = list(ub.compress(edges, list(subgraph.has_edges(edges))))
            subgraph.remove_edges_from(prev_edges)
            infr.review_graphs[UNREV].add_edges_from(prev_edges)

        infr.pos_redun_nids.clear()
        infr.neg_redun_metagraph.clear()
        infr.nid_to_errors.clear()

        if __debug__:
            infr.assert_disjoint_invariant()

    def clear_edges(infr):
        """
        Removes all edges from the graph
        """
        for graph in infr.review_graphs.values():
            graph.remove_edges_from(list(graph.edges()))
        infr.graph.remove_edges_from(list(infr.graph.edges()))
        infr.pos_redun_nids.clear()
        infr.neg_redun_metagraph.clear()
        infr.nid_to_errors.clear()

    def reset(infr, state='empty'):
        """
        Removes all edges from graph and resets name labels.

        Example:
            >>> from graphid.core.annot_inference import *  # NOQA
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(num_pccs=5)
            >>> assert len(list(infr.edges())) > 0
            >>> infr.reset(state='empty')
            >>> assert len(list(infr.edges())) == 0
        """
        infr.clear_edges()
        infr.clear_feedback()
        if state == 'empty':
            # Remove all edges, and component names
            infr.clear_name_labels()
        elif state == 'orig':
            raise NotImplementedError('unused')
            infr.reset_name_labels()
        else:
            raise ValueError('Unknown state=%r' % (state,))

    def reset_name_labels(infr):
        """ Resets all annotation node name labels to their initial values """
        infr.print('reset_name_labels', 1)
        orig_names = infr.get_node_attrs('orig_name_label')
        infr.set_node_attrs('name_label', orig_names)

    def clear_name_labels(infr):
        """ Sets all annotation node name labels to be unknown """
        infr.print('clear_name_labels()', 1)
        # make distinct names for all nodes
        distinct_names = {
            node: -aid for node, aid in infr.get_node_attrs('aid').items()
        }
        infr.set_node_attrs('name_label', distinct_names)


class NameRelabel:

    def node_label(infr, aid):
        return infr.pos_graph.node_label(aid)

    def node_labels(infr, *aids):
        return infr.pos_graph.node_labels(*aids)

    def _next_nid(infr):
        if getattr(infr, 'nid_counter', None) is None:
            nids = nx.get_node_attributes(infr.graph, 'name_label')
            infr.nid_counter = max(nids)
        infr.nid_counter += 1
        new_nid = infr.nid_counter
        return new_nid

    def _rectify_names(infr, old_names, new_labels):
        """
        Finds the best assignment of old names based on the new groups each is
        assigned to.

        old_names  = [None, None, None, 1, 2, 3, 3, 4, 4, 4, 5, None]
        new_labels = [   1,    2,    2, 3, 4, 5, 5, 6, 3, 3, 7, 7]
        """
        infr.print('rectifying name lists', 3)
        newlabel_to_oldnames = ub.group_items(old_names, new_labels)
        unique_newlabels = list(newlabel_to_oldnames.keys())
        grouped_oldnames_ = list(ub.take(newlabel_to_oldnames, unique_newlabels))
        # Mark annots that are unknown and still grouped by themselves
        still_unknown = [len(g) == 1 and g[0] is None for g in grouped_oldnames_]
        # Remove nones for name rectifier
        grouped_oldnames = [
            [n for n in oldgroup if n is not None]
            for oldgroup in grouped_oldnames_]
        new_names = util.name_rectifier.find_consistent_labeling(
            grouped_oldnames, verbose=infr.verbose >= 3, extra_prefix=None)

        unknown_labels = list(ub.compress(unique_newlabels, still_unknown))

        new_flags = [n is None for n in new_names]
        label_to_name = ub.dzip(unique_newlabels, new_names)
        needs_assign = list(ub.compress(unique_newlabels, new_flags))
        return label_to_name, needs_assign, unknown_labels

    def _rectified_relabel(infr, cc_subgraphs):
        """
        Reuses as many names as possible
        """
        # Determine which names can be reused
        infr.print('grouping names for rectification', 3)
        grouped_oldnames_ = [
            list(nx.get_node_attributes(subgraph, 'name_label').values())
            for count, subgraph in enumerate(cc_subgraphs)
        ]
        # Make sure negatives dont get priority
        grouped_oldnames = [
            [n for n in group if len(group) == 1 or n > 0]
            for group in grouped_oldnames_
        ]
        infr.print('begin rectification of %d grouped old names' % (
            len(grouped_oldnames)), 2)
        new_labels = util.name_rectifier.find_consistent_labeling(
            grouped_oldnames, verbose=infr.verbose >= 3)
        infr.print('done rectifying new names', 2)
        new_flags = [
            not isinstance(n, int) and n.startswith('_extra_name')
            for n in new_labels
        ]

        for idx in util.where(new_flags):
            new_labels[idx] = infr._next_nid()

        for idx, label in enumerate(new_labels):
            if label < 0 and len(grouped_oldnames[idx]) > 1:
                # Remove negative ids for grouped items
                new_labels[idx] = infr._next_nid()
        return new_labels

    def relabel_using_reviews(infr, graph=None, rectify=True):
        """
        Relabels nodes in graph based on positive connected components

        This will change the 'name_label' of the nodes to be consistent while
        preserving any existing names as best as possible. If rectify=False,
        this will be faster, but the old names may not be preserved and each
        PCC will be assigned an arbitrary name.

        Note:
            if something messes up you can call infr.reset_labels_to_ibeis() to
            reset node labels to their original values --- this will almost
            always put the graph in an inconsistent state --- but then you can
            this with rectify=True to fix everything up.

        Args:
            graph (nx.Graph, optional): only edges in `graph` are relabeled
                defaults to current graph.
            rectify (bool, optional): if True names attempt to remain
                consistent otherwise there are no restrictions on name labels
                other than that they are distinct.

        Example:
            >>> from graphid import demo, util
            >>> infr = demo.demodata_infr(num_pccs=5, pos_redun=1)
            >>> names0 = set(infr.get_node_attrs('name_label').values())
            >>> infr.relabel_using_reviews(rectify=True)
            >>> names1 = set(infr.get_node_attrs('name_label').values())
            >>> assert names0 == names1
            >>> # wont change because its the entire graph
            >>> #infr.relabel_using_reviews(rectify=False)
            >>> #names2 = set(infr.get_node_attrs('name_label').values())

        """
        infr.print('relabel_using_reviews', 2)
        if graph is None:
            graph = infr.graph

        # Get subgraphs and check consistency
        cc_subgraphs = []
        num_inconsistent = 0
        for cc in infr.positive_components(graph=graph):
            cc_subgraphs.append(infr.graph.subgraph(cc))
            if not infr.is_consistent(cc):
                num_inconsistent += 1

        infr.print('num_inconsistent = %r' % (num_inconsistent,), 2)
        if infr.verbose >= 2:
            cc_sizes = list(map(len, cc_subgraphs))
            pcc_size_hist = ub.dict_hist(cc_sizes)
            pcc_size_stats = util.stats_dict(cc_sizes)
            if len(pcc_size_hist) < 8:
                infr.print('PCC size hist = %s' % (ub.urepr(pcc_size_hist),))
            infr.print('PCC size stats = %s' % (ub.urepr(pcc_size_stats),))

        if rectify:
            # Rectified relabeling, preserves grouping and labeling if possible
            new_labels = infr._rectified_relabel(cc_subgraphs)
        else:
            # Arbitrary relabeling, only preserves grouping
            if graph is infr.graph:
                # Use union find labels
                new_labels = {
                    count:
                    infr.node_label(next(iter(subgraph.nodes())))
                    for count, subgraph in enumerate(cc_subgraphs)
                }
            else:
                new_labels = {count: infr._next_nid()
                              for count, subgraph in enumerate(cc_subgraphs)}

        for count, subgraph in enumerate(cc_subgraphs):
            new_nid = new_labels[count]
            node_to_newlabel = ub.dzip(subgraph.nodes(), [new_nid])
            infr.set_node_attrs('name_label', node_to_newlabel)

        num_names = len(cc_subgraphs)
        infr.print('done relabeling', 3)
        return num_names, num_inconsistent


class MiscHelpers:

    def _rectify_nids(infr, aids, nids):
        if aids is None:
            raise ValueError('aids cannot be None')
        if nids is None:
            nids = [-aid for aid in aids]
        elif not ub.iterable(nids):
            nids = [nids] * len(aids)
        return nids

    def remove_aids(infr, aids):
        """
        Remove annotations from the graph.

        Returns:
            dict: split: indicates which PCCs were split by this action.

        Note:
            This may cause unintended splits!

        CommandLine:
            xdoctest -m graphid.core.annot_inference MiscHelpers.remove_aids

        Example:
            >>> from graphid import demo, util
            >>> infr = demo.demodata_infr(num_pccs=5, pos_redun=1)
            >>> infr.refresh_candidate_edges()
            >>> infr.pin_node_layout()
            >>> before = infr.copy()
            >>> aids = infr.aids[::5]
            >>> splits = infr.remove_aids(aids)
            >>> assert len(splits['old']) > 0
            >>> infr.assert_invariants()
            >>> # xdoctest: +REQUIRES(--show)
            >>> util.autompl()
            >>> after = infr
            >>> before.show(fnum=1, pnum=(1, 2, 1), pickable=True)
            >>> after.show(fnum=1, pnum=(1, 2, 2), pickable=True)
        """
        infr.print('remove_aids len(aids)={}'.format(len(aids)), level=3)

        # Determine which edges are going to be removed
        remove_edges = util.edges_outgoing(infr.graph, aids)

        old_groups = list(infr.positive_components())

        # Remove from tertiary bookkeeping structures
        remove_idxs = list(ub.take(util.make_index_lookup(infr.aids), aids))
        util.delete_items_by_index(infr.orig_name_labels, remove_idxs)
        util.delete_items_by_index(infr.aids, remove_idxs)
        infr.aids_set = set(infr.aids)

        # Remove from secondary bookkeeping structures
        util.delete_dict_keys(infr.external_feedback, remove_edges)
        util.delete_dict_keys(infr.internal_feedback, remove_edges)

        # Remove from core bookkeeping structures
        infr.graph.remove_nodes_from(aids)
        for graph in infr.review_graphs.values():
            graph.remove_nodes_from(aids)

        infr.queue.delete_items(remove_edges)

        # TODO: should refactor to preform a dyanmic step, but in this case is
        # less work to use a bazooka to shoot a fly.
        infr.apply_nondynamic_update()

        # I'm unsure if relabeling is necessary
        infr.relabel_using_reviews()

        new_groups = list(infr.positive_components())

        # print('old_groups = {!r}'.format(old_groups))
        # print('new_groups = {!r}'.format(new_groups))
        delta = util.grouping_delta(old_groups, new_groups)
        splits = delta['splits']

        n_old = len(splits['old'])
        n_new = len(list(ub.flatten(splits['new'])))
        infr.print(
            'removing {} aids split {} old PCCs into {} new PCCs'.format(
                len(aids), n_old, n_new))

        return splits
        # print(ub.urepr(delta, nl=2))

    def add_aids(infr, aids, nids=None):
        """
        CommandLine:
            python -m graphid.core.annot_inference MiscHelpers.add_aids

        Doctest:
            >>> aids_ = [1, 2, 3, 4, 5, 6, 7, 9]
            >>> infr = AnnotInference(aids=aids_, autoinit=True)
            >>> aids = [2, 22, 7, 9, 8]
            >>> nids = None
            >>> infr.add_aids(aids, nids)
            >>> result = infr.aids
            >>> print(result)
            >>> assert len(infr.graph) == len(infr.aids)
            [1, 2, 3, 4, 5, 6, 7, 9, 22, 8]
        """
        if aids is None:
            raise ValueError('aids cannot be None')
        nids = infr._rectify_nids(aids, nids)
        assert len(aids) == len(nids), 'must correspond'
        if infr.aids is None:
            nids = infr._rectify_nids(aids, nids)
            # Set object attributes
            infr.aids = aids
            infr.aids_set = set(infr.aids)
            infr.orig_name_labels = nids
        else:
            aid_to_idx = util.make_index_lookup(infr.aids)
            orig_idxs = list(ub.take(aid_to_idx, aids, None))
            new_flags = util.flag_None_items(orig_idxs)
            new_aids = list(ub.compress(aids, new_flags))
            new_nids = list(ub.compress(nids, new_flags))
            # Extend object attributes
            infr.aids.extend(new_aids)
            infr.orig_name_labels.extend(new_nids)
            infr.aids_set.update(new_aids)
            infr.update_node_attributes(new_aids, new_nids)

            if infr.graph is not None:
                infr.graph.add_nodes_from(aids)
                for subgraph in infr.review_graphs.values():
                    subgraph.add_nodes_from(aids)
            nids = set(infr.pos_graph.node_labels(*aids))
            infr.neg_metagraph.add_nodes_from(nids)

    def update_node_attributes(infr, aids=None, nids=None):
        if aids is None:
            aids = infr.aids
            nids = infr.orig_name_labels
        assert aids is not None, 'must have aids'
        assert nids is not None, 'must have nids'
        node_to_aid = {aid: aid for aid in aids}
        node_to_nid = {aid: nid for aid, nid in zip(aids, nids)}
        assert len(node_to_nid) == len(node_to_aid)

        infr.graph.add_nodes_from(aids)
        for subgraph in infr.review_graphs.values():
            subgraph.add_nodes_from(aids)

        infr.set_node_attrs('aid', node_to_aid)
        infr.set_node_attrs('name_label', node_to_nid)
        infr.set_node_attrs('orig_name_label', node_to_nid)
        # TODO: depricate these, they will always be identity I think
        # this is also taken care of by relabel_using_reviews

    def initialize_graph(infr, graph=None):
        """
        Constructs the internal networkx Graph objects
        """
        infr.print('initialize_graph', 1)
        if graph is None:
            infr.graph = infr._graph_cls()
        else:
            infr.graph = graph

        infr.review_graphs[POSTV] = util.DynConnGraph()
        infr.review_graphs[NEGTV] = infr._graph_cls()
        infr.review_graphs[INCMP] = infr._graph_cls()
        infr.review_graphs[UNKWN] = infr._graph_cls()
        infr.review_graphs[UNREV] = infr._graph_cls()

        if graph is not None:
            for u, v, d in graph.edges(data=True):
                evidence_decision = d.get('evidence_decision', UNREV)
                meta_decision = d.get('meta_decision', NULL)
                decision = _rectify_decision(evidence_decision, meta_decision)
                if decision in {POSTV, NEGTV, INCMP, UNREV, UNKWN}:
                    infr.review_graphs[decision].add_edge(u, v)
                else:
                    raise ValueError('Unknown decision=%r' % (decision,))

        infr.update_node_attributes()

    def print(infr, msg, level=1, color=None):
        if color is None:
            color = 'turquoise' if ub.WIN32 else 'blue'

        RECORD_LOGS = True
        INFER_CALLER = True

        if INFER_CALLER:
            from xdoctest.dynamic_analysis import get_parent_frame
            # Record the name of the calling function
            parent_name = get_parent_frame().f_code.co_name
            msg = '[{}] '.format(parent_name) + msg

        if RECORD_LOGS:
            # Append the message to an internal log deque
            infr.logs.append((msg, color))
            if len(infr.logs) == infr.logs.maxlen:
                infr.log_index = max(infr.log_index - 1, 0)

        if infr.verbose >= level:
            # Print the message to stdout
            loglevel = logging.INFO
            util.cprint('[infr] ' + msg, color)
        else:
            loglevel = logging.DEBUG
        if infr.logger:
            # Send the message to a python logger
            infr.logger.log(loglevel, msg)

    def latest_logs(infr, colored=False):
        index = infr.log_index
        infr.log_index = len(infr.logs)
        if colored:
            return [infr.logs[x] for x in range(index, len(infr.logs))]
        else:
            return [infr.logs[x][0] for x in range(index, len(infr.logs))]

    def dump_logs(infr):
        print('--- <LOG DUMP> ---')
        for msg, color in infr.logs:
            util.cprint('[infr] ' + msg, color)
        print(r'--- </LOG DUMP> ---')


class AltConstructors:
    _graph_cls = util.NiceGraph
    # _graph_cls = nx.Graph
    # _graph_cls = nx.DiGraph

    @classmethod
    def from_pairs(AnnotInference, aid_pairs, attrs=None, verbose=False):
        G = AnnotInference._graph_cls()
        if any(a1 == a2 for a1, a2 in aid_pairs):
            raise AssertionError('cannot have self-edges')
        G.add_edges_from(aid_pairs)
        if attrs is not None:
            for key in attrs.keys():
                values = ub.dzip(aid_pairs, attrs[key])
                nx.set_edge_attributes(G, name=key, values=values)
        infr = AnnotInference.from_netx(G, verbose=verbose)
        return infr

    @classmethod
    def from_netx(AnnotInference, G, verbose=False, infer=True):
        """
        Creates an AnnotInference object from a networkx graph
        """
        aids = list(G.nodes())
        nids = [-a for a in aids]
        infr = AnnotInference(aids, nids, autoinit=False,
                              verbose=verbose)
        infr.initialize_graph(graph=G)
        # hack
        orig_name_labels = [infr.pos_graph.node_label(a) for a in aids]
        infr.orig_name_labels = orig_name_labels
        infr.set_node_attrs('orig_name_label', ub.dzip(aids, orig_name_labels))
        if infer:
            infr.apply_nondynamic_update()
        return infr

    def status(infr, extended=False):
        """
        Returns information about the state of the graph.

        Args:
            extended (bool): if True, adds in extra information that requires
                an O(|E|) amount of computation, otherwise only O(1) stats that
                are dynamically tracked are returned.

        Returns:
            dict: a dictionary containing status information. Each of the keys
                represents the following information:

                    nNodes: number of nodes in the graph
                    nEdges: number of edges in the graph
                    nCCs: number of positive connected components
                    nPostvEdges: number of edges labeled as positive
                    nNegtvEdges: number of edges labeled as negative
                    nIncmpEdges: number of edges labeled as incomparable
                    nUnrevEdges: number of edges labeled as unreviewed
                    nPosRedunCCs: the number of PCCs which are currently
                        k-positive-redundant, i.e. we are confident those
                        PCCs are the same individual.
                    nNegRedunPairs: the number of PCCs pairs which are
                        currently k-negative-redundant, i.e. we are confident
                        those PCCs are different individuals.
                    nInconsistentCCs: the number of inconsistent PCCs that need
                        to be fixed, i.e. the number of PCCs with an internal
                        negative edges.

                If extended is True, then the following keys are also present
                    nNegEdgesWithin: number of negatives edges inside PCCs
                    nNegEdgesBetween: number of negative edges between PCCs
                    nIncompEdgesWithin: number of incomparable edges inside PCCs
                    nIncompEdgesBetween: number of incomparable edges between PCCs
                    nUnrevEdgesWithin: number of unreviewed edges inside PCCs
                    nUrevEdgesBetween: number of unreviewed edges between PCCs

        Example:
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(num_pccs=5, p_incon=0.5, pcc_size=10)
            >>> print(ub.urepr(infr.status(extended=True)))
            {
                'nNodes': 50,
                'nEdges': 93,
                'nCCs': 5,
                'nPostvEdges': 66,
                'nNegtvEdges': 10,
                'nIncmpEdges': 2,
                'nUnrevEdges': 15,
                'nPosRedunCCs': 1,
                'nNegRedunPairs': 2,
                'nInconsistentCCs': 3,
                'nNegEdgesWithin': 4,
                'nNegEdgesBetween': 6,
                'nIncompEdgesWithin': 0,
                'nIncompEdgesBetween': 2,
                'nUnrevEdgesWithin': 15,
                'nUrevEdgesBetween': 0,
            }
        """
        status_dict = ub.odict([
            ('nNodes', len(infr.aids)),
            ('nEdges', infr.graph.number_of_edges()),
            ('nCCs', infr.pos_graph.number_of_components()),
            ('nPostvEdges', infr.pos_graph.number_of_edges()),
            ('nNegtvEdges', infr.neg_graph.number_of_edges()),
            ('nIncmpEdges', infr.incomp_graph.number_of_edges()),
            ('nUnrevEdges', infr.unreviewed_graph.number_of_edges()),
            ('nPosRedunCCs', len(infr.pos_redun_nids)),
            ('nNegRedunPairs', infr.neg_redun_metagraph.number_of_edges()),
            ('nInconsistentCCs', len(infr.nid_to_errors)),
            #('nUnkwnEdges', infr.unknown_graph.number_of_edges()),
        ])
        if extended:
            def count_within_between(edges):
                n_within = 0
                n_between = 0
                for u, v in edges:
                    nid1, nid2 = infr.pos_graph.node_labels(u, v)
                    if nid1 == nid2:
                        n_within += 1
                    else:
                        n_between += 1
                return n_within, n_between

            a, b = count_within_between(infr.neg_graph.edges())
            status_dict['nNegEdgesWithin'] = a
            status_dict['nNegEdgesBetween'] = b

            a, b = count_within_between(infr.incomp_graph.edges())
            status_dict['nIncompEdgesWithin'] = a
            status_dict['nIncompEdgesBetween'] = b

            a, b = count_within_between(infr.unreviewed_graph.edges())
            status_dict['nUnrevEdgesWithin'] = a
            status_dict['nUrevEdgesBetween'] = b

        return status_dict


class AnnotInference(ub.NiceRepr,
                     # Old internal stuffs
                     AltConstructors,
                     MiscHelpers,
                     Feedback,
                     NameRelabel,
                     Consistency,
                     # New annot_inference algorithm stuffs
                     mixin_dynamic.NonDynamicUpdate,
                     mixin_dynamic.Recovery,
                     mixin_dynamic.DynamicUpdate,
                     mixin_redundancy.Redundancy,
                     mixin_priority.Priority,
                     # General helpers
                     mixin_invariants.AssertInvariants,
                     mixin_helpers.DummyEdges,
                     mixin_helpers.Convenience,
                     mixin_helpers.AttrAccess,
                     # Simulation and Loops
                     mixin_simulation.SimulationHelpers,
                     mixin_loops.InfrReviewers,
                     mixin_loops.InfrLoops,
                     # For matching and candidate edge callbacks
                     mixin_callbacks.InfrCallbacks,
                     mixin_callbacks.InfrCandidates,
                     # Visualization
                     mixin_viz.GraphVisualization,
                     ):
    """
    class for maintaining state of an identification

    CommandLine:
        python -m graphid.core.annot_inference AnnotInference
        python -m graphid.core.annot_inference AnnotInference --show

    Example:
        >>> from graphid.core import AnnotInference
        >>> import pytest
        >>> infr = AnnotInference()
        >>> print('infr = {}'.format(infr))
        infr = <AnnotInference(nNodes=0, nEdges=0, nCCs=0)>
        >>> infr.add_aids(list(range(1, 6)))
        >>> print('infr = {}'.format(infr))
        infr = <AnnotInference(nNodes=5, nEdges=0, nCCs=5)>
        >>> # Add some feedback
        >>> infr.params['allow_unseen_nodes'] = False
        >>> infr.add_feedback((1, 2), POSTV)
        >>> infr.add_feedback((1, 3), INCMP)
        >>> infr.add_feedback((1, 4), NEGTV)
        >>> with pytest.raises(ValueError):
        >>>     infr.add_feedback((1, 10), NEGTV)
        >>> with pytest.raises(ValueError):
        >>>     infr.add_feedback((11, 12), NEGTV)
        >>> print('infr = {}'.format(infr))
        infr = <AnnotInference(nNodes=5, nEdges=3, nCCs=4)>
        >>> # xdoctest: +REQUIRES(--show)
        >>> infr.show_graph()
        >>> util.show_if_requested()
    """

    def __getstate__(self):
        state = self.__dict__.copy()
        # Dont pickle generators
        state['_gen'] = None
        state['logger'] = None
        return state

    def __init__(infr, aids=[], nids=None, autoinit=True, verbose=False):
        """
        Ignore:
            pass
        """
        # infr.verbose = verbose
        infr.name = None
        infr.verbose = verbose

        # setup logging
        infr.logger = None

        infr.logs = collections.deque(maxlen=10000)
        infr.log_index = 0

        # If not dirty, new feedback should dynamically maintain a consistent
        # state. If dirty it means we need to recompute connected compoments
        # before we can continue with dynamic review.
        infr.dirty = False
        infr.readonly = False

        infr.aids = None
        infr.aids_set = None
        infr.orig_name_labels = None

        # Underlying graph structure
        infr.graph = None
        infr.review_graphs = {
            POSTV: None,
            NEGTV: None,
            INCMP: None,
            UNKWN: None,
            UNREV: None,
        }

        # Criterion
        infr.queue = util.PriorityQueue()
        infr.refresh = None

        infr.review_counter = it.count(0)
        infr.nid_counter = None

        # Dynamic Properties (requires bookkeeping)
        infr.nid_to_errors = {}

        # Recover graph holds positive edges of inconsistent PCCs
        infr.recover_graph = util.DynConnGraph()
        # Set of PCCs that are positive redundant
        infr.pos_redun_nids = set([])
        # Represents the metagraph of negative edges between PCCs
        infr.neg_redun_metagraph = infr._graph_cls()
        # NEW VERSION: metagraph of PCCs with ANY number of negative edges
        # between them. The weight on the edge should represent the strength.
        infr.neg_metagraph = infr._graph_cls()

        # This should represent The feedback read from a database. We do not
        # need to do any updates to an external database based on this data.
        infr.external_feedback = ub.ddict(list)
        # Feedback that has not been synced with the external database.
        # Once we sync, this is merged into external feedback.
        infr.internal_feedback = ub.ddict(list)

        # Bookkeeping
        infr.edge_truth = {}
        infr.task_probs = ub.ddict(dict)

        # A generator that maintains the state of the algorithm
        infr._gen = None

        # Computer vision algorithms
        infr.ranker = None  # the ranking algorithm (e.g. LNBNN)
        infr.verifier = None  # the match_state classifier
        infr.verifiers = None  # dictionary of tasks -> classifier

        # TODO: move to params
        infr.task_thresh = {
            'match_state': {
                POSTV: np.inf,
                NEGTV: np.inf,
                INCMP: np.inf,
            },
            'photobomb_state': {
                'pb': np.inf,
                'nopb': np.inf,
            }
        }

        # Parameters / Configurations / Callbacks
        infr.callbacks = {
            'request_review': None,
            'review_ready': None,
            'review_finished': None,
        }

        infr.params = {
            # If False, adding edges with non-existant nodes will error
            # Otherwise it will silently add them.
            'allow_unseen_nodes': True,

            'manual.n_peek': 1,

            'ranking.enabled': True,
            'ranking.ntop': 5,

            'algo.max_outer_loops': None,
            'algo.quickstart': False,
            'algo.hardcase': False,

            # Dynamic Inference
            'inference.enabled': True,
            'inference.update_attrs': True,

            # Termination / Refresh
            'refresh.window': 20,
            'refresh.patience': 72,
            'refresh.thresh': 0.052,
            'refresh.method': 'binomial',

            # Redundancy
            # if redun.enabled is True, then redundant edges will be ignored by
            # # the priority queue and extra edges needed to achieve minimum
            # redundancy will be searched for if the queue is empty.
            'redun.enabled': True,
            # positive/negative k
            'redun.pos': 2,
            'redun.neg': 2,
            # does positive/negative augmentation
            'redun.enforce_pos': True,
            'redun.enforce_neg': True,
            # prevents user interaction in final phase
            'redun.neg.only_auto': True,

            # Only review CCs connected by confidence less than this value
            # a good values is 'pretty_sure'
            'queue.conf.thresh': None,

            # Autoreviewer params
            'autoreview.enabled': True,
            'autoreview.prioritize_nonpos': True,
        }

        infr._viz_image_config = {
            'in_image': False,
            'thumbsize': 221,
        }

        infr.verifier_params = {}  # TODO
        infr.ranker_params = {}

        # Developer modes (consoldate this)
        infr.test_mode = False
        infr.simulation_mode = False

        # set to the current phase of the main loop
        # (mostly for testing)
        infr.loop_phase = None

        # Testing state
        infr.metrics_list = None
        infr.test_state = None
        infr.test_gt_pos_graph = None
        infr.nid_to_gt_cc = None
        infr.node_truth = None
        infr.real_n_pcc_mst_edges = None

        # External: Can we remove these?
        infr.cm_list = None
        infr.vsone_matches = {}
        infr.qreq_ = None
        infr.manual_wgt = None

        infr.print('__init__', level=1)
        infr.add_aids(aids, nids)
        if autoinit:
            infr.initialize_graph()
            if isinstance(autoinit, str):
                raise Exception('Cannot autoinit this way anymore')

    def __nice__(infr):
        """ for ub.NiceRepr """
        if infr.graph is None:
            return 'nAids=%r, G=None' % (len(infr.aids))
        else:
            fmt = 'nNodes={}, nEdges={}, nCCs={}'
            msg = fmt.format(
                len(infr.aids),
                infr.graph.number_of_edges(),
                infr.pos_graph.number_of_components(),
                # infr.incomp_graph.number_of_edges(),
                # infr.unreviewed_graph.number_of_edges(),
            )
            return msg

    def subparams(infr, prefix):
        """
        Returns dict of params prefixed with <prefix>.
        The returned dict does not contain the prefix

        Example:
            >>> infr = AnnotInference()
            >>> result = ub.urepr(infr.subparams('refresh'), nl=0, precision=1, sort=1)
            >>> print(result)
            {'method': 'binomial', 'patience': 72, 'thresh': 0.1, 'window': 20}
        """
        prefix_ = prefix + '.'
        subparams = {k[len(prefix_):]: v for k, v in infr.params.items()
                     if k.startswith(prefix_)}
        return subparams

    def copy(infr):
        infr2 = AnnotInference(
            copy.deepcopy(infr.aids),
            copy.deepcopy(infr.orig_name_labels), autoinit=False,
            verbose=infr.verbose)

        # shallow algorithm classes
        infr2.verifiers = infr.verifiers
        infr2.ranker = infr.ranker

        infr2.graph = infr.graph.copy()
        infr2.external_feedback = copy.deepcopy(infr.external_feedback)
        infr2.internal_feedback = copy.deepcopy(infr.internal_feedback)
        infr2.cm_list = copy.deepcopy(infr.cm_list)
        infr2.qreq_ = copy.deepcopy(infr.qreq_)
        infr2.nid_counter = infr.nid_counter

        infr2.recover_graph = copy.deepcopy(infr.recover_graph)

        infr2.pos_redun_nids = copy.deepcopy(infr.pos_redun_nids)
        infr2.neg_redun_metagraph = copy.deepcopy(infr.neg_redun_metagraph)
        infr2.neg_metagraph = copy.deepcopy(infr.neg_metagraph)

        infr2.review_graphs = copy.deepcopy(infr.review_graphs)
        infr2.nid_to_errors = copy.deepcopy(infr.nid_to_errors)

        infr2.readonly = infr.readonly
        infr2.dirty = infr.dirty

        infr2.test_mode = infr.test_mode
        infr2.test_mode = infr.test_mode
        infr2.simulation_mode = infr.simulation_mode

        infr.queue = copy.deepcopy(infr.queue)

        infr.params = copy.deepcopy(infr.params)
        infr2._viz_image_config = infr._viz_image_config.copy()

        if infr.test_mode:
            infr2.test_state = copy.deepcopy(infr.test_state)
            infr2.metrics_list = copy.deepcopy(infr.metrics_list)
        return infr2

    def subgraph(infr, aids):
        """
        Makes a new inference object that is a subset of the original.

        Note, this is not robust, be careful. The subgraph should be treated as
        read only. Do not commit any reviews made from here.
        """
        orig_name_labels = list(infr.gen_node_values('orig_name_label', aids))
        infr2 = AnnotInference(aids, orig_name_labels,
                               autoinit=False, verbose=infr.verbose)
        # deep copy the graph structure
        infr2.graph = infr.graph.subgraph(aids).copy()
        infr2.readonly = True
        infr2.verifiers = infr.verifiers
        infr2.ranker = infr.ranker

        infr.params = copy.deepcopy(infr.params)
        infr2._viz_image_config = infr._viz_image_config.copy()

        # infr2._viz_init_nodes = infr._viz_image_config
        # infr2._viz_image_config_dirty = infr._viz_image_config_dirty
        infr2.edge_truth = {
            e: infr.edge_truth[e] for e in infr2.graph.edges()
            if e in infr.edge_truth
        }

        # TODO: internal/external feedback

        infr2.nid_counter = infr.nid_counter
        infr2.dirty = True
        infr2.cm_list = None
        infr2.qreq_ = None

        # TODO:
        # infr2.nid_to_errors {}  # = copy.deepcopy(infr.nid_to_errors)
        # infr2.recover_graph = copy.deepcopy(infr.recover_graph)
        # infr2.pos_redun_nids = copy.deepcopy(infr.pos_redun_nids)
        # infr2.neg_redun_metagraph = copy.deepcopy(infr.neg_redun_metagraph)

        infr2.review_graphs = {}
        for k, g in infr.review_graphs.items():
            if g is None:
                infr2.review_graphs[k] = None
            elif k == POSTV:
                infr2.review_graphs[k] = g.subgraph(aids, dynamic=True)
            else:
                infr2.review_graphs[k] = g.subgraph(aids)
        return infr2

    def set_config(infr, config, **kw):
        pass


if __name__ == '__main__':
    """
    CommandLine:
        python -m graphid.core.annot_inference all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
