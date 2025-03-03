"""
This file handles dynamically updating the graph state based on new feedback.
This involves handling lots of different cases, which can get confusing (it
confuses me and I wrote it). To better understand the dynamic case a good first
step would be to understand the nondynamic case defined by
`apply_nondynamic_update`. This function automatically puts the graph into a
state that satisfies the dynamic invariants. Any dynamic operation followed by
a call to this function should be a no-op, which you can used to check if a
dynamic operation is implemented correctly.


TODO:
    Negative bookkeeping, needs a small re-organization fix.
    MOVE FROM neg_redun_metagraph TO neg_metagraph

    Instead of maintaining a graph that contains PCCS which are neg redundant
    to each other, the graph should maintain PCCs that have ANY negative edge
    between them (aka 1 neg redundant). Then that edge should store a flag
    indicating the strength / redundancy of that connection.
    A better idea might be to store both neg_redun_metagraph AND neg_metagraph.

    TODO: this (all neg-redun functionality can be easilly consolidated into
    the neg-metagraph-update. note, we have to allow inconsistent pccs to be in
    the neg redun graph, we just filter them out afterwords)

"""
import ubelt as ub
import numpy as np
import networkx as nx
from functools import partial
from graphid.core import state as const
from graphid.util import nx_utils as nxu
from graphid.core.state import (POSTV, NEGTV, INCMP, UNREV, UNKWN, UNINFERABLE)
from graphid.core.state import (SAME, DIFF, NULL)  # NOQA


class DynamicUpdate:
    """
    # 12 total possible states

    # details of these states.
    POSITIVE, WITHIN, CONSISTENT
        * pos-within never changes PCC status
        * never introduces inconsistency
        * might add pos-redun
    POSITIVE, WITHIN, INCONSISTENT
        * pos-within never changes PCC status
        * might fix inconsistent edge
    POSITIVE, BETWEEN, BOTH_CONSISTENT
        * pos-between edge always does merge
    POSITIVE, BETWEEN, ANY_INCONSISTENT
        * pos-between edge always does merge
        * pos-between never fixes inconsistency

    NEGATIVE, WITHIN, CONSISTENT
        * might split PCC, results will be consistent
        * might causes an inconsistency
    NEGATIVE, WITHIN, INCONSISTENT
        * might split PCC, results may be inconsistent
    NEGATIVE, BETWEEN, BOTH_CONSISTENT
        * might add neg-redun
    NEGATIVE, BETWEEN, ANY_INCONSISTENT
        * might add to incon-neg-external
        * neg-redun not tracked for incon.

    UNINFERABLE, WITHIN, CONSISTENT
        * might remove pos-redun
        * might split PCC, results will be consistent
    UNINFERABLE, WITHIN, INCONSISTENT
        * might split PCC, results may be inconsistent
    UNINFERABLE, BETWEEN, BOTH_CONSISTENT
        * might remove neg-redun
    UNINFERABLE, BETWEEN, ANY_INCONSISTENT
        * might remove incon-neg-external
    """

    def ensure_edges_from(infr, edges):
        """
        Finds edges that don't exist and adds them as unreviwed edges.
        Returns new edges that were added.
        """
        edges = list(edges)
        # Only add edges that don't exist
        new_edges = [e for e in edges if not infr.has_edge(e)]
        infr.graph.add_edges_from(new_edges,
                                  evidence_decision=UNREV,
                                  meta_decision=UNREV,
                                  decision=UNREV,
                                  num_reviews=0)
        # No inference is needed by expliclty creating unreviewed edges that
        # already implicitly existsed.
        infr._add_review_edges_from(new_edges, decision=UNREV)
        return new_edges

    def _add_review_edges_from(infr, edges, decision=UNREV):
        infr.print('add {} edges decision={}'.format(len(edges), decision), 1)
        # Add to review graph corresponding to decision
        infr.review_graphs[decision].add_edges_from(edges)
        # Remove from previously existing graphs
        for k, G in infr.review_graphs.items():
            if k != decision:
                G.remove_edges_from(edges)

    def _add_review_edge(infr, edge, decision):
        """
        Adds an edge to the appropriate data structure
        """
        # infr.print('add review edge=%r, decision=%r' % (edge, decision), 20)
        # Add to review graph corresponding to decision
        infr.review_graphs[decision].add_edge(*edge)
        # Remove from previously existing graphs
        for k, G in infr.review_graphs.items():
            if k != decision:
                if G.has_edge(*edge):
                    G.remove_edge(*edge)

    def _get_current_decision(infr, edge):
        """
        Find if any data structure has the edge
        """
        for decision, G in infr.review_graphs.items():
            if G.has_edge(*edge):
                return decision
        return UNREV

    def on_between(infr, edge, decision, prev_decision, nid1, nid2,
                   merge_nid=None):
        """
        Callback when a review is made between two PCCs
        """
        action = ['between']

        infr._update_neg_metagraph(
            decision, prev_decision, nid1, nid2, merge_nid=merge_nid)

        if merge_nid is not None:
            # A merge occurred
            if infr.params['inference.update_attrs']:
                cc = infr.pos_graph.component(merge_nid)
                infr.set_node_attrs('name_label', ub.dzip(cc, [merge_nid]))
            # FIXME: this state is ugly
            action += ['merge']
        else:
            if decision == NEGTV:
                action += ['neg-evidence']
            elif decision == INCMP:
                action += ['incomp-evidence']
            else:
                action += ['other-evidence']
        return action

    def on_within(infr, edge, decision, prev_decision, nid, split_nids=None):
        """
        Callback when a review is made inside a PCC

        Args:
            edge: the edge reviewed
            decision: the new decision
            prev_decision: the old decision
            nid: the old nid the edge is inside of
            split_nids: the tuple of new nids created if this decision splits a PCC
        """
        action = ['within']

        infr._update_neg_metagraph(
            decision, prev_decision, nid, nid, split_nids=split_nids)

        if split_nids is not None:
            # A split occurred
            if infr.params['inference.update_attrs']:
                new_nid1, new_nid2 = split_nids
                cc1 = infr.pos_graph.component(new_nid1)
                cc2 = infr.pos_graph.component(new_nid2)
                infr.set_node_attrs('name_label', ub.dzip(cc1, [new_nid1]))
                infr.set_node_attrs('name_label', ub.dzip(cc2, [new_nid2]))
            action += ['split']
        else:
            if decision == POSTV:
                action += ['pos-evidence']
            elif decision == INCMP:
                action += ['incomp-evidence']
            elif decision == NEGTV:
                action += ['neg-evidence']
            else:
                action += ['other-evidence']
        return action

    def _update_neg_metagraph(infr, decision, prev_decision, nid1, nid2,
                              merge_nid=None, split_nids=None):
        """
        Update the negative metagraph based a new review

        TODO:
            we can likely consolidate lots of neg_redun_metagraph
            functionality into this function. Just check when the
            weights are above or under the threshold and update
            accordingly.
        """
        nmg = infr.neg_metagraph

        if decision == NEGTV and prev_decision != NEGTV:
            # New negative feedback. Add meta edge or increase weight
            if not nmg.has_edge(nid1, nid2):
                nmg.add_edge(nid1, nid2, weight=1)
            else:
                nmg.edges[nid1, nid2]['weight'] += 1
        elif decision != NEGTV and prev_decision == NEGTV:
            # Undid negative feedback. Remove meta edge or decrease weight.
            nmg.edges[nid1, nid2]['weight'] -= 1
            if nmg.edges[nid1, nid2]['weight'] == 0:
                nmg.remove_edge(nid1, nid2)

        if merge_nid:
            # Combine the negative edges between the merged PCCS
            assert split_nids is None
            # Find external nids marked as negative
            prev_edges = nmg.edges(nbunch=[nid1, nid2], data=True)
            # Map external neg edges onto new merged PCC
            # Accumulate weights between duplicate new name edges
            lookup = {nid1: merge_nid, nid2: merge_nid}
            ne_accum = {}
            for (u, v, d) in prev_edges:
                new_ne = infr.e_(lookup.get(u, u), lookup.get(v, v))
                if new_ne in ne_accum:
                    ne_accum[new_ne]['weight'] += d['weight']
                else:
                    ne_accum[new_ne] = d
            merged_edges = ((u, v, d) for (u, v), d in ne_accum.items())

            nmg.remove_nodes_from([nid1, nid2])
            nmg.add_node(merge_nid)
            nmg.add_edges_from(merged_edges)

        if split_nids:
            # Splitup the negative edges between the split PCCS
            assert merge_nid is None
            assert nid1 == nid2
            old_nid = nid1

            # Find the nodes we need to check against
            extern_nids = set(nmg.neighbors(old_nid))
            if old_nid in extern_nids:
                extern_nids.remove(old_nid)
                extern_nids.update(split_nids)

            # Determine how to split existing negative edges between the split
            # by going back to the original negative graph.
            split_edges = []
            for new_nid in split_nids:
                cc1 = infr.pos_graph.component(new_nid)
                for other_nid in extern_nids:
                    cc2 = infr.pos_graph.component(other_nid)
                    num = sum(1 for _ in nxu.edges_between(
                        infr.neg_graph, cc1, cc2, assume_dense=False))
                    if num:
                        split_edges.append(
                            (new_nid, other_nid, {'weight': num}))

            nmg.remove_node(old_nid)
            nmg.add_nodes_from(split_nids)
            nmg.add_edges_from(split_edges)

    def _positive_decision(infr, edge):
        """
        Logic for a dynamic positive decision.  A positive decision is evidence
        that two annots should be in the same PCC

        Note, this could be an incomparable edge, but with a meta_decision of
        same.

        Ignore:
            >>> from graphid import demo
            >>> kwargs = dict(num_pccs=3, p_incon=0, size=100)
            >>> infr = demo.demodata_infr(infer=False, **kwargs)
            >>> infr.apply_nondynamic_update()
            >>> cc1 = next(infr.positive_components())

            %timeit list(infr.pos_graph.subgraph(cc1, dynamic=True).edges())
            %timeit list(infr.pos_graph.subgraph(cc1, dynamic=False).edges())
            %timeit list(nxu.edges_inside(infr.pos_graph, cc1))
        """
        decision = POSTV
        nid1, nid2 = infr.pos_graph.node_labels(*edge)
        incon1, incon2 = infr.recover_graph.has_nodes(edge)
        all_consistent = not (incon1 or incon2)
        was_within = nid1 == nid2

        print_ = partial(infr.print, level=4)
        prev_decision = infr._get_current_decision(edge)

        if was_within:
            infr._add_review_edge(edge, decision)
            if all_consistent:
                print_('pos-within-clean')
                infr.update_pos_redun(nid1, may_remove=False)
            else:
                print_('pos-within-dirty')
                infr._check_inconsistency(nid1)
            action = infr.on_within(edge, decision, prev_decision, nid1, None)
        else:
            # print_('Merge case')
            cc1 = infr.pos_graph.component(nid1)
            cc2 = infr.pos_graph.component(nid2)

            if not all_consistent:
                # We are merging PCCs that are not all consistent
                # This will keep us in a dirty state.
                print_('pos-between-dirty-merge')
                if not incon1:
                    recover_edges = list(nxu.edges_inside(infr.pos_graph, cc1))
                else:
                    recover_edges = list(nxu.edges_inside(infr.pos_graph, cc2))
                infr.recover_graph.add_edges_from(recover_edges)
                infr._purge_redun_flags(nid1)
                infr._purge_redun_flags(nid2)
                infr._add_review_edge(edge, decision)
                infr.recover_graph.add_edge(*edge)
                new_nid = infr.pos_graph.node_label(edge[0])
                # purge and re-add the inconsistency
                # (Note: the following three lines were added to fix
                #  a neg_meta_graph test, and may not be the best way to do it)
                infr._purge_error_edges(nid1)
                infr._purge_error_edges(nid2)
                infr._new_inconsistency(new_nid)
            elif any(nxu.edges_cross(infr.neg_graph, cc1, cc2)):
                # There are negative edges bridging these PCCS
                # this will put the graph into a dirty (inconsistent) state.
                print_('pos-between-clean-merge-dirty')
                infr._purge_redun_flags(nid1)
                infr._purge_redun_flags(nid2)
                infr._add_review_edge(edge, decision)
                new_nid = infr.pos_graph.node_label(edge[0])
                infr._new_inconsistency(new_nid)
            else:
                # We are merging two clean PCCs, everything is good
                print_('pos-between-clean-merge-clean')
                infr._purge_redun_flags(nid1)
                infr._purge_redun_flags(nid2)
                infr._add_review_edge(edge, decision)
                new_nid = infr.pos_graph.node_label(edge[0])
                infr.update_extern_neg_redun(new_nid, may_remove=False)
                infr.update_pos_redun(new_nid, may_remove=False)
            action = infr.on_between(edge, decision, prev_decision, nid1, nid2,
                                     merge_nid=new_nid)
        return action

    def _negative_decision(infr, edge):
        """
        Logic for a dynamic negative decision.  A negative decision is evidence
        that two annots should not be in the same PCC
        """
        decision = NEGTV
        nid1, nid2 = infr.node_labels(*edge)
        incon1, incon2 = infr.recover_graph.has_nodes(edge)
        all_consistent = not (incon1 or incon2)
        prev_decision = infr._get_current_decision(edge)

        infr._add_review_edge(edge, decision)
        new_nid1, new_nid2 = infr.pos_graph.node_labels(*edge)

        was_within = nid1 == nid2
        was_split = was_within and new_nid1 != new_nid2

        print_ = partial(infr.print, level=4)

        if was_within:
            if was_split:
                if all_consistent:
                    print_('neg-within-split-clean')
                    prev_neg_nids = infr._purge_redun_flags(nid1)
                    infr.update_neg_redun_to(new_nid1, prev_neg_nids)
                    infr.update_neg_redun_to(new_nid2, prev_neg_nids)
                    infr.update_neg_redun_to(new_nid1, [new_nid2])
                    # infr.update_extern_neg_redun(new_nid1, may_remove=False)
                    # infr.update_extern_neg_redun(new_nid2, may_remove=False)
                    infr.update_pos_redun(new_nid1, may_remove=False)
                    infr.update_pos_redun(new_nid2, may_remove=False)
                else:
                    print_('neg-within-split-dirty')
                    if infr.recover_graph.has_edge(*edge):
                        infr.recover_graph.remove_edge(*edge)
                    infr._purge_error_edges(nid1)
                    infr._purge_redun_flags(nid1)
                    infr._check_inconsistency(new_nid1)
                    infr._check_inconsistency(new_nid2)
                # Signal that a split occurred
                action = infr.on_within(edge, decision, prev_decision, nid1,
                                        split_nids=(new_nid1, new_nid2))
            else:
                if all_consistent:
                    print_('neg-within-clean')
                    infr._purge_redun_flags(new_nid1)
                    infr._new_inconsistency(new_nid1)
                else:
                    print_('neg-within-dirty')
                    infr._check_inconsistency(new_nid1)
                action = infr.on_within(edge, decision, prev_decision, new_nid1)
        else:
            if all_consistent:
                print_('neg-between-clean')
                infr.update_neg_redun_to(new_nid1, [new_nid2], may_remove=False)
            else:
                print_('neg-between-dirty')
                # nothing to do if a negative edge is added between two PCCs
                # where at least one is inconsistent
                pass
            action = infr.on_between(edge, decision, prev_decision, new_nid1,
                                     new_nid2)
        return action

    def _uninferable_decision(infr, edge, decision):
        """
        Logic for a dynamic uninferable negative decision An uninferrable
        decision does not provide any evidence about PCC status and is either:
            incomparable, unreviewed, or unknown
        """
        nid1, nid2 = infr.pos_graph.node_labels(*edge)
        incon1 = infr.recover_graph.has_node(edge[0])
        incon2 = infr.recover_graph.has_node(edge[1])
        all_consistent = not (incon1 or incon2)

        was_within = nid1 == nid2
        prev_decision = infr._get_current_decision(edge)

        print_ = partial(infr.print, level=4)

        try:
            prefix = {INCMP: 'incmp', UNREV: 'unrev',
                      UNKWN: 'unkown'}[decision]
        except KeyError:
            raise KeyError('decision can only be UNREV, INCMP, or UNKWN')

        infr._add_review_edge(edge, decision)

        if was_within:
            if prev_decision == POSTV:
                # changed an existing positive edge
                if infr.recover_graph.has_edge(*edge):
                    infr.recover_graph.remove_edge(*edge)
                new_nid1, new_nid2 = infr.pos_graph.node_labels(*edge)
                was_split = new_nid1 != new_nid2
                if was_split:
                    old_nid = nid1
                    prev_neg_nids = infr._purge_redun_flags(old_nid)
                    if all_consistent:
                        print_('%s-within-pos-split-clean' % prefix)
                        # split case
                        infr.update_neg_redun_to(new_nid1, prev_neg_nids)
                        infr.update_neg_redun_to(new_nid2, prev_neg_nids)
                        # for other_nid in prev_neg_nids:
                        #     infr.update_neg_redun_to(new_nid1, [other_nid])
                        #     infr.update_neg_redun_to(new_nid2, [other_nid])
                        infr.update_neg_redun_to(new_nid1, [new_nid2])
                        infr.update_pos_redun(new_nid1, may_remove=False)
                        infr.update_pos_redun(new_nid2, may_remove=False)
                    else:
                        print_('%s-within-pos-split-dirty' % prefix)
                        if infr.recover_graph.has_edge(*edge):
                            infr.recover_graph.remove_edge(*edge)
                        infr._purge_error_edges(nid1)
                        infr._check_inconsistency(new_nid1)
                        infr._check_inconsistency(new_nid2)
                    # Signal that a split occurred
                    action = infr.on_within(edge, decision, prev_decision,
                                            nid1, split_nids=(new_nid1,
                                                              new_nid2))
                else:
                    if all_consistent:
                        print_('%s-within-pos-clean' % prefix)
                        infr.update_pos_redun(new_nid1, may_add=False)
                    else:
                        print_('%s-within-pos-dirty' % prefix)
                        # Overwriting a positive edge that is not a split
                        # in an inconsistent component, means no inference.
                    action = infr.on_within(edge, decision, prev_decision,
                                            new_nid1)
            elif prev_decision == NEGTV:
                print_('%s-within-neg-dirty' % prefix)
                assert not all_consistent
                infr._check_inconsistency(nid1)
                action = infr.on_within(edge, decision, prev_decision, nid1)
            else:
                if all_consistent:
                    print_('%s-within-clean' % prefix)
                else:
                    print_('%s-within-dirty' % prefix)
                action = infr.on_within(edge, decision, prev_decision, nid1)
        else:
            if prev_decision == NEGTV:
                if all_consistent:
                    # changed and existing negative edge only influences
                    # consistent pairs of PCCs
                    print_('incon-between-neg-clean')
                    infr.update_neg_redun_to(nid1, [nid2], may_add=False)
                else:
                    print_('incon-between-neg-dirty')
            else:
                print_('incon-between')
                # HACK, this sortof fixes inferred state not being set
                if infr.params['inference.update_attrs']:
                    if decision == INCMP:
                        pass
                        # if not infr.is_neg_redundant(cc1, cc2, k=1):
                        #     # TODO: verify that there isn't a negative inferred
                        #     # state
                        #     infr.set_edge_attrs(
                        #         'inferred_state', ub.dzip([edge], [INCMP])
                        #     )
            action = infr.on_between(edge, decision, prev_decision, nid1, nid2)
        return action


class Recovery:
    """ recovery funcs """

    def is_recovering(infr, edge=None):
        """
        Checks to see if the graph is inconsinsistent.

        Args:
            edge (None): If None, then returns True if the graph contains any
                inconsistency. Otherwise, returns True if the edge is related
                to an inconsistent component via a positive or negative
                connection.

        Returns:
            bool: flag

        CommandLine:
            python -m graphid.core.mixin_dynamic Recovery.is_recovering

        Doctest:
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(num_pccs=4, size=4, ignore_pair=True)
            >>> infr.ensure_cliques(meta_decision=SAME)
            >>> a, b, c, d = map(list, infr.positive_components())
            >>> assert infr.is_recovering() is False
            >>> infr.add_feedback((a[0], a[1]), NEGTV)
            >>> assert infr.is_recovering() is True
            >>> assert infr.is_recovering((a[2], a[3])) is True
            >>> assert infr.is_recovering((a[3], b[0])) is True
            >>> assert infr.is_recovering((b[0], b[1])) is False
            >>> infr.add_feedback((a[3], b[2]), NEGTV)
            >>> assert infr.is_recovering((b[0], b[1])) is True
            >>> assert infr.is_recovering((c[0], d[0])) is False
            >>> infr.add_feedback((b[2], c[0]), NEGTV)
            >>> assert infr.is_recovering((c[0], d[0])) is False
            >>> result = ub.urepr({
            >>>     'iccs': list(infr.inconsistent_components()),
            >>>     'pccs': sorted([cc for cc in infr.positive_components()], key=min),
            >>> }, nobr=1, sorted=True, si=True, itemsep='', sep='', nl=1)
            >>> print(result)
            iccs: [{1,2,3,4}],
            pccs: [{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}],
        """
        if len(infr.recover_graph) == 0:
            # We can short-circuit if there is no inconsistency
            return False
        if edge is None:
            # By the short circuit we know the graph is inconsistent
            return True
        for nid in set(infr.node_labels(*edge)):
            # Is this edge part of a CC that has an error?
            if nid in infr.nid_to_errors:
                return True
            # Is this edge connected to a CC that has an error?
            cc = infr.pos_graph.component(nid)
            for nid2 in infr.find_neg_nids_to(cc):
                if nid2 in infr.nid_to_errors:
                    return True
        # If none of these conditions are true we are far enough away from the
        # inconsistency to ignore it.
        return False

    def _purge_error_edges(infr, nid):
        """
        Removes all error edges associated with a PCC so they can be recomputed
        or resolved.
        """
        old_error_edges = infr.nid_to_errors.pop(nid, [])
        # Remove priority from old error edges
        if infr.params['inference.update_attrs']:
            infr.set_edge_attrs('maybe_error',
                                ub.dzip(old_error_edges, [None]))
        infr._remove_edge_priority(old_error_edges)
        was_clean = len(old_error_edges) > 0
        return was_clean

    def _set_error_edges(infr, nid, new_error_edges):
        # flag error edges
        infr.nid_to_errors[nid] = new_error_edges
        # choose one and give it insanely high priority
        if infr.params['inference.update_attrs']:
            infr.set_edge_attrs('maybe_error',
                                ub.dzip(new_error_edges, [True]))
        infr._increase_priority(new_error_edges, 10)

    def maybe_error_edges(infr):
        return ub.flatten(infr.nid_to_errors.values())

    def _new_inconsistency(infr, nid):
        cc = infr.pos_graph.component(nid)
        pos_edges = infr.pos_graph.edges(cc)
        infr.recover_graph.add_edges_from(pos_edges)
        num = infr.recover_graph.number_of_components()
        msg = 'New inconsistency {} total'.format(num)
        infr.print(msg, 2, color='red')
        infr._check_inconsistency(nid, cc=cc)

    def _check_inconsistency(infr, nid, cc=None):
        """
        Check if a PCC contains an error
        """
        if cc is None:
            cc = infr.pos_graph.component(nid)
        was_clean = infr._purge_error_edges(nid)
        neg_edges = list(nxu.edges_inside(infr.neg_graph, cc))
        if neg_edges:
            pos_subgraph_ = infr.pos_graph.subgraph(cc, dynamic=False).copy()
            if not nx.is_connected(pos_subgraph_):
                print('cc = %r' % (cc,))
                print('pos_subgraph_ = %r' % (pos_subgraph_,))
                raise AssertionError('must be connected')
            hypothesis = dict(infr.hypothesis_errors(pos_subgraph_, neg_edges))
            assert len(hypothesis) > 0, 'must have at least one'
            infr._set_error_edges(nid, set(hypothesis.keys()))
            is_clean = False
        else:
            infr.recover_graph.remove_nodes_from(cc)
            num = infr.recover_graph.number_of_components()
            # num = len(list(nx.connected_components(infr.recover_graph)))
            msg = ('An inconsistent PCC recovered, '
                   '{} inconsistent PCC(s) remain').format(num)
            infr.print(msg, 2, color='green')
            infr.update_pos_redun(nid, force=True)
            infr.update_extern_neg_redun(nid, force=True)
            is_clean = True
        return (was_clean, is_clean)

    def _mincut_edge_weights(infr, edges_):
        conf_gen = infr.gen_edge_values('confidence', edges_,
                                        default='unspecified')
        conf_gen = ['unspecified' if c is None else c for c in conf_gen]
        code_to_conf = const.CONFIDENCE.CODE_TO_INT
        code_to_conf = {
            'absolutely_sure' : 4.0,
            'pretty_sure'     : 0.6,
            'not_sure'        : 0.2,
            'guessing'        : 0.0,
            'unspecified'     : 0.0,
        }
        confs = np.array(list(ub.take(code_to_conf, conf_gen)))
        # confs = np.array([0 if c is None else c for c in confs])

        prob_gen = infr.gen_edge_values('prob_match', edges_, default=0)
        probs = np.array(list(prob_gen))

        nrev_gen = infr.gen_edge_values('num_reviews', edges_, default=0)
        nrev = np.array(list(nrev_gen))

        weight = nrev + probs + confs
        return weight

    def hypothesis_errors(infr, pos_subgraph, neg_edges):
        if not nx.is_connected(pos_subgraph):
            raise AssertionError('Not connected' + repr(pos_subgraph))
        infr.print(
            'Find hypothesis errors in {} nodes with {} neg edges'.format(
                len(pos_subgraph), len(neg_edges)), 3)

        pos_edges = list(pos_subgraph.edges())

        neg_weight = infr._mincut_edge_weights(neg_edges)
        pos_weight = infr._mincut_edge_weights(pos_edges)

        capacity = 'weight'
        nx.set_edge_attributes(pos_subgraph, name=capacity, values=ub.dzip(pos_edges, pos_weight))

        # Solve a multicut problem for multiple pairs of terminal nodes.
        # Running multiple min-cuts produces a k-factor approximation
        maybe_error_edges = set([])
        for (s, t), join_weight in zip(neg_edges, neg_weight):
            cut_weight, parts = nx.minimum_cut(pos_subgraph, s, t,
                                               capacity=capacity)
            cut_edgeset = nxu.edges_cross(pos_subgraph, *parts)
            if join_weight < cut_weight:
                join_edgeset = {(s, t)}
                chosen = join_edgeset
                hypothesis = POSTV
            else:
                chosen = cut_edgeset
                hypothesis = NEGTV
            for edge in chosen:
                if edge not in maybe_error_edges:
                    maybe_error_edges.add(edge)
                    yield (edge, hypothesis)


class NonDynamicUpdate:

    def apply_nondynamic_update(infr, graph=None):
        """
        Recomputes all dynamic bookkeeping for a graph in any state.
        This ensures that subsequent dyanmic inference can be applied.

        Example:
            >>> from graphid import demo
            >>> num_pccs = 250
            >>> kwargs = dict(num_pccs=100, p_incon=.3)
            >>> infr = demo.demodata_infr(infer=False, **kwargs)
            >>> graph = None
            >>> infr.apply_nondynamic_update()
            >>> infr.assert_neg_metagraph()
        """
        # Cluster edges by category
        ne_to_edges = infr.collapsed_meta_edges()
        categories = infr.categorize_edges(graph, ne_to_edges)

        infr.set_edge_attrs(
            'inferred_state',
            ub.dzip(ub.flatten(categories[POSTV].values()), ['same'])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ub.dzip(ub.flatten(categories[NEGTV].values()), ['diff'])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ub.dzip(ub.flatten(categories[INCMP].values()), [INCMP])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ub.dzip(ub.flatten(categories[UNKWN].values()), [UNKWN])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ub.dzip(ub.flatten(categories[UNREV].values()), [None])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ub.dzip(ub.flatten(categories['inconsistent_internal'].values()),
                    ['inconsistent_internal'])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ub.dzip(ub.flatten(categories['inconsistent_external'].values()),
                    ['inconsistent_external'])
        )

        # Ensure bookkeeping is taken care of
        # * positive redundancy
        # * negative redundancy
        # * inconsistency
        infr.pos_redun_nids = set(infr.find_pos_redun_nids())
        infr.neg_redun_metagraph = infr._graph_cls(list(infr.find_neg_redun_nids()))

        # make a node for each PCC, and place an edge between any pccs with at
        # least one negative edge, with weight being the number of negative
        # edges. Self loops indicate inconsistency.
        infr.neg_metagraph = infr._graph_cls()
        infr.neg_metagraph.add_nodes_from(infr.pos_graph.component_labels())
        for (nid1, nid2), edges in ne_to_edges[NEGTV].items():
            infr.neg_metagraph.add_edge(nid1, nid2, weight=len(edges))

        infr.recover_graph.clear()
        nid_to_errors = {}
        for nid, intern_edges in categories['inconsistent_internal'].items():
            cc = infr.pos_graph.component_nodes(nid)
            pos_subgraph = infr.pos_graph.subgraph(cc, dynamic=False).copy()
            neg_edges = list(nxu.edges_inside(infr.neg_graph, cc))
            recover_hypothesis = dict(infr.hypothesis_errors(pos_subgraph,
                                                             neg_edges))
            nid_to_errors[nid] = set(recover_hypothesis.keys())
            infr.recover_graph.add_edges_from(pos_subgraph.edges())

        # Delete old hypothesis
        infr.set_edge_attrs(
            'maybe_error',
            ub.dzip(ub.flatten(infr.nid_to_errors.values()), [None])
        )
        # Set new hypothesis
        infr.set_edge_attrs(
            'maybe_error',
            ub.dzip(ub.flatten(nid_to_errors.values()), [True])
        )
        infr.nid_to_errors = nid_to_errors

        # no longer dirty
        if graph is None:
            infr.dirty = False

    def collapsed_meta_edges(infr, graph=None):
        """
        Collapse the grah such that each PCC is a node. Get a list of edges
        within/between each PCC.
        """
        states = (POSTV, NEGTV, INCMP, UNREV, UNKWN)
        rev_graph = {key: infr.review_graphs[key] for key in states}
        if graph is None or graph is infr.graph:
            graph = infr.graph
            nodes = None
        else:
            # Need to extract relevant subgraphs
            nodes = list(graph.nodes())
            for key in states:
                if key == POSTV:
                    rev_graph[key] = rev_graph[key].subgraph(nodes,
                                                             dynamic=False)
                else:
                    rev_graph[key] = rev_graph[key].subgraph(nodes)

        # TODO: Rebalance union find to ensure parents is a single lookup
        # infr.pos_graph._union_find.rebalance(nodes)
        # node_to_label = infr.pos_graph._union_find.parents
        node_to_label = infr.pos_graph._union_find

        # Get reviewed edges using fast lookup structures
        ne_to_edges = {
            key: nxu.group_name_edges(rev_graph[key], node_to_label)
            for key in states
        }
        return ne_to_edges

    def categorize_edges(infr, graph=None, ne_to_edges=None):
        """
        Non-dynamically computes the status of each edge in the graph.
        This is can be used to verify the dynamic computations and update when
        the dynamic state is lost.

        Example:
            >>> from graphid import demo
            >>> num_pccs = 250 if ub.argflag('--profile') else 100
            >>> kwargs = dict(num_pccs=100, p_incon=.3)
            >>> infr = demo.demodata_infr(infer=False, **kwargs)
            >>> graph = None
            >>> cat = infr.categorize_edges()
        """
        states = (POSTV, NEGTV, INCMP, UNREV, UNKWN)

        if ne_to_edges is None:
            ne_to_edges = infr.collapsed_meta_edges(graph)

        # Use reviewed edges to determine status of PCCs (repr by name ids)
        # The next steps will rectify duplicates in these sets
        name_edges = {key: set(ne_to_edges[key].keys()) for key in states}

        # Positive and negative decisions override incomparable and unreviewed
        for key in UNINFERABLE:
            name_edges[key].difference_update(name_edges[POSTV])
            name_edges[key].difference_update(name_edges[NEGTV])

        # Negative edges within a PCC signals that an inconsistency exists
        # Remove inconsistencies from the name edges
        incon_internal_ne = name_edges[NEGTV].intersection(name_edges[POSTV])
        name_edges[POSTV].difference_update(incon_internal_ne)
        name_edges[NEGTV].difference_update(incon_internal_ne)

        if __debug__:
            assert all(n1 == n2 for n1, n2 in name_edges[POSTV]), (
                'All positive edges should be internal to a PCC')
            assert len(name_edges[INCMP].intersection(incon_internal_ne)) == 0
            assert len(name_edges[UNREV].intersection(incon_internal_ne)) == 0
            assert len(name_edges[UNKWN].intersection(incon_internal_ne)) == 0
            assert all(n1 == n2 for n1, n2 in incon_internal_ne), (
                'incon_internal edges should be internal to a PCC')

        # External inconsistentices are edges leaving inconsistent components
        incon_internal_nids = {n1 for n1, n2 in incon_internal_ne}
        incon_external_ne = set([])
        # Find all edges leaving an inconsistent PCC
        for key in (NEGTV,) + UNINFERABLE:
            incon_external_ne.update({
                (nid1, nid2) for nid1, nid2 in name_edges[key]
                if nid1 in incon_internal_nids or nid2 in incon_internal_nids
            })
        for key in (NEGTV,) + UNINFERABLE:
            name_edges[key].difference_update(incon_external_ne)

        # Inference between names is now complete.
        # Now we expand this inference and project the labels onto the
        # annotation edges corresponding to each name edge.

        # Version of union that accepts generators
        union = lambda gen: set.union(*gen)  # NOQA

        # Find edges within consistent PCCs
        positive = {
            nid1: union(
                ne_to_edges[key][(nid1, nid2)]
                for key in (POSTV,) + UNINFERABLE)
            for nid1, nid2 in name_edges[POSTV]
        }
        # Find edges between 1-negative-redundant consistent PCCs
        negative = {
            (nid1, nid2): union(
                ne_to_edges[key][(nid1, nid2)]
                for key in (NEGTV,) + UNINFERABLE)
            for nid1, nid2 in name_edges[NEGTV]
        }
        # Find edges internal to inconsistent PCCs
        incon_internal = {
            nid: union(
                ne_to_edges[key][(nid, nid)]
                for key in (POSTV, NEGTV,) + UNINFERABLE)
            for nid in incon_internal_nids
        }
        # Find edges leaving inconsistent PCCs
        incon_external = {
            (nid1, nid2): union(
                ne_to_edges[key][(nid1, nid2)]
                for key in (NEGTV,) + UNINFERABLE)
            for nid1, nid2 in incon_external_ne
        }
        # Unknown names may have been comparable but the reviewer did not
        # know and could not guess. Likely bad quality.
        unknown = {
            (nid1, nid2): ne_to_edges[UNKWN][(nid1, nid2)]
            for (nid1, nid2) in name_edges[UNKWN]
        }
        # Incomparable names cannot make inference about any other edges
        notcomparable = {
            (nid1, nid2): ne_to_edges[INCMP][(nid1, nid2)]
            for (nid1, nid2) in name_edges[INCMP]
        }
        # Unreviewed edges are between any name not known to be negative
        # (this ignores specific incomparable edges)
        unreviewed = {
            (nid1, nid2): ne_to_edges[UNREV][(nid1, nid2)]
            for (nid1, nid2) in name_edges[UNREV]
        }

        ne_categories = {
            POSTV: positive,
            NEGTV: negative,
            UNREV: unreviewed,
            INCMP: notcomparable,
            UNKWN: unknown,
            'inconsistent_internal': incon_internal,
            'inconsistent_external': incon_external,
        }
        return ne_categories

if __name__ == '__main__':
    """
    CommandLine:
        python -m graphid.core.mixin_dynamic all
        python ~/code/graphid/graphid/ibeis/mixin_matching.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
