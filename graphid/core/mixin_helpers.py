import itertools as it
import networkx as nx
import operator
import numpy as np
import ubelt as ub
import pandas as pd
from graphid import util
from graphid.core import state as const
from graphid.core.state import POSTV, NEGTV, INCMP, UNREV, UNKWN
from graphid.core.state import SAME, DIFF, NULL  # NOQA
from graphid.util import nx_utils as nxu
from graphid.util.nx_utils import e_


class AttrAccess:
    """ Contains non-core helper functions """

    def gen_node_attrs(infr, key, nodes=None, default=ub.NoParam):
        return util.nx_gen_node_attrs(
                infr.graph, key, nodes=nodes, default=default)

    def gen_edge_attrs(infr, key, edges=None, default=ub.NoParam,
                       on_missing=None):
        """ maybe change to gen edge items """
        return util.nx_gen_edge_attrs(
                infr.graph, key, edges=edges, default=default,
                on_missing=on_missing)

    def gen_node_values(infr, key, nodes, default=ub.NoParam):
        return util.nx_gen_node_values(
            infr.graph, key, nodes, default=default)

    def gen_edge_values(infr, key, edges=None, default=ub.NoParam,
                        on_missing='error', on_keyerr='default'):
        return util.nx_gen_edge_values(
            infr.graph, key, edges, default=default, on_missing=on_missing,
            on_keyerr=on_keyerr)

    def get_node_attrs(infr, key, nodes=None, default=ub.NoParam):
        """ Networkx node getter helper """
        return dict(infr.gen_node_attrs(key, nodes=nodes, default=default))

    def get_edge_attrs(infr, key, edges=None, default=ub.NoParam,
                       on_missing=None):
        """ Networkx edge getter helper """
        return dict(infr.gen_edge_attrs(key, edges=edges, default=default,
                                        on_missing=on_missing))

    def _get_edges_where(infr, key, op, val, edges=None, default=ub.NoParam,
                         on_missing=None):
        edge_to_attr = infr.gen_edge_attrs(key, edges=edges, default=default,
                                           on_missing=on_missing)
        return (e for e, v in edge_to_attr if op(v, val))

    def get_edges_where_eq(infr, key, val, edges=None, default=ub.NoParam,
                           on_missing=None):
        return infr._get_edges_where(key, operator.eq, val, edges=edges,
                                     default=default, on_missing=on_missing)

    def get_edges_where_ne(infr, key, val, edges=None, default=ub.NoParam,
                           on_missing=None):
        return infr._get_edges_where(key, operator.ne, val, edges=edges,
                                     default=default, on_missing=on_missing)

    def set_node_attrs(infr, key, node_to_prop):
        """ Networkx node setter helper """
        return nx.set_node_attributes(infr.graph, name=key, values=node_to_prop)

    def set_edge_attrs(infr, key, edge_to_prop):
        """ Networkx edge setter helper """
        return nx.set_edge_attributes(infr.graph, name=key, values=edge_to_prop)

    def get_edge_attr(infr, edge, key, default=ub.NoParam, on_missing='error'):
        """ single edge getter helper """
        return infr.get_edge_attrs(key, [edge], default=default,
                                   on_missing=on_missing)[edge]

    def set_edge_attr(infr, edge, attr):
        """ single edge setter helper """
        for key, value in attr.items():
            infr.set_edge_attrs(key, {edge: value})

    def get_annot_attrs(infr, key, aids):
        """ Wrapper around get_node_attrs specific to annotation nodes """
        attr_list = list(infr.get_node_attrs(key, aids).values())
        return attr_list

    def edges(infr, data=False):
        if data:
            return ((e_(u, v), d) for u, v, d in infr.graph.edges(data=True))
        else:
            return (e_(u, v) for u, v in infr.graph.edges())

    def has_edge(infr, edge):
        return infr.graph.has_edge(*edge)
        # redge = edge[::-1]
        # flag = infr.graph.has_edge(*edge) or infr.graph.has_edge(*redge)
        # return flag

    def get_edge_data(infr, edge):
        return infr.graph.get_edge_data(*edge)

    def get_nonvisual_edge_data(infr, edge, on_missing='filter'):
        data = infr.get_edge_data(edge)
        if data is not None:
            data = util.delete_dict_keys(data.copy(), infr.visual_edge_attrs)
        else:
            if on_missing == 'filter':
                data = None
            elif on_missing == 'default':
                data = {}
            elif on_missing == 'error':
                raise KeyError('graph does not have edge %r ' % (edge,))
        return data

    def get_edge_dataframe(infr, edges=None, all=False):
        if edges is None:
            edges = infr.edges()
        edge_datas = {e: infr.get_nonvisual_edge_data(e) for e in edges}
        edge_datas = {e: {k: None for k in infr.feedback_data_keys}
                      if d is None else d for e, d in edge_datas.items()}
        edge_df = pd.DataFrame.from_dict(edge_datas, orient='index')

        part = ['evidence_decision', 'meta_decision', 'tags', 'user_id']
        neworder = util.partial_order(edge_df.columns, part)
        if hasattr(edge_df, 'reindex_axis'):
            edge_df = edge_df.reindex_axis(neworder, axis=1)
        else:
            edge_df = edge_df.reindex(neworder, axis=1)
        if not all:
            todrop = ['review_id', 'timestamp', 'timestamp_s1', 'timestamp_c2',
                      'timestamp_c1']
            todrop = [c for c in todrop if c in edge_df.columns]
            edge_df = edge_df.drop(todrop, axis=1)
        # pd.DataFrame.from_dict(edge_datas, orient='list')
        return edge_df

    def get_edge_df_text(infr, edges=None, highlight=True):
        df = infr.get_edge_dataframe(edges)
        df_str = df.to_string()
        if highlight:
            df_str = util.highlight_regex(df_str, util.regex_word(SAME), color='blue')
            df_str = util.highlight_regex(df_str, util.regex_word(POSTV), color='blue')
            df_str = util.highlight_regex(df_str, util.regex_word(DIFF), color='red')
            df_str = util.highlight_regex(df_str, util.regex_word(NEGTV), color='red')
            df_str = util.highlight_regex(df_str, util.regex_word(INCMP), color='yellow')
        return df_str


class Convenience:
    @staticmethod
    def e_(u, v):
        return e_(u, v)

    @property
    def pos_graph(infr):
        return infr.review_graphs[POSTV]

    @property
    def neg_graph(infr):
        return infr.review_graphs[NEGTV]

    @property
    def incomp_graph(infr):
        return infr.review_graphs[INCMP]

    @property
    def unreviewed_graph(infr):
        return infr.review_graphs[UNREV]

    @property
    def unknown_graph(infr):
        return infr.review_graphs[UNKWN]

    def print_graph_info(infr):
        print(ub.urepr(util.graph_info(infr.simplify_graph())))

    def print_graph_connections(infr, label='orig_name_label'):
        """
        label = 'orig_name_label'
        """
        node_to_label = infr.get_node_attrs(label)
        label_to_nodes = ub.group_items(node_to_label.keys(),
                                        node_to_label.values())
        print('CC info')
        for name, cc in label_to_nodes.items():
            print('\nname = %r' % (name,))
            edges = list(nxu.edges_between(infr.graph, cc))
            print(infr.get_edge_df_text(edges))

        print('CC pair info')
        for (n1, cc1), (n2, cc2) in it.combinations(label_to_nodes.items(), 2):
            if n1 == n2:
                continue
            print('\nname_pair = {}-vs-{}'.format(n1, n2))
            edges = list(nxu.edges_between(infr.graph, cc1, cc2))
            print(infr.get_edge_df_text(edges))

    def print_within_connection_info(infr, edge=None, cc=None, aid=None, nid=None):
        if edge is not None:
            aid, aid2 = edge
        if nid is not None:
            cc = infr.pos_graph._ccs[nid]
        if aid is not None:
            cc = infr.pos_graph.connected_to(aid)
        # subgraph = infr.graph.subgraph(cc)
        # list(nxu.complement_edges(subgraph))
        edges = list(nxu.edges_between(infr.graph, cc))
        print(infr.get_edge_df_text(edges))

    def pair_connection_info(infr, aid1, aid2):
        """
        Helps debugging when ibs.nids has info that annotmatch/staging do not
        Note: the relevant ibs parts were removed. Perhaps this is not useful
        now or should be moved to the ibeis plugin?

        Example:
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(num_pccs=3, size=4)
            >>> aid1, aid2 = 1, 2
            >>> print(infr.pair_connection_info(aid1, aid2))
        """

        nid1, nid2 = infr.pos_graph.node_labels(aid1, aid2)
        cc1 = infr.pos_graph.connected_to(aid1)
        cc2 = infr.pos_graph.connected_to(aid2)

        # First check directly relationships

        def get_aug_df(edges):
            df = infr.get_edge_dataframe(edges)
            if len(df):
                df.index.names = ('aid1', 'aid2')
                nids = np.array([
                    infr.pos_graph.node_labels(u, v)
                    for u, v in list(df.index)])
                df = df.assign(nid1=nids.T[0], nid2=nids.T[1])
                part = ['nid1', 'nid2', 'evidence_decision', 'tags', 'user_id']
                neworder = util.partial_order(df.columns, part)
                if hasattr(df, 'reindex_axis'):
                    df = df.reindex_axis(neworder, axis=1)
                else:
                    df = df.reindex(neworder, axis=1)
                todrop = [c for c in ['review_id', 'timestamp']
                          if c in df.columns]
                df = df.drop(todrop, axis=1)
            return df

        def print_df(df, lbl):
            df_str = df.to_string()
            df_str = util.highlight_regex(df_str, util.regex_word(str(aid1)), color='blue')
            df_str = util.highlight_regex(df_str, util.regex_word(str(aid2)), color='red')
            if nid1 not in {aid1, aid2}:
                df_str = util.highlight_regex(df_str, util.regex_word(str(nid1)), color='darkblue')
            if nid2 not in {aid1, aid2}:
                df_str = util.highlight_regex(df_str, util.regex_word(str(nid2)), color='darkred')
            print('\n\n=====')
            print(lbl)
            print('=====')
            print(df_str)

        print('================')
        print('Pair Connection Info')
        print('================')

        # ibs = infr.ibs
        # nid1_, nid2_ = ibs.get_annot_nids([aid1, aid2])
        print('AIDS        aid1, aid2 = %r, %r' % (aid1, aid2))
        # print('INFR NAMES: nid1, nid2 = %r, %r' % (nid1, nid2))
        if nid1 == nid2:
            print('INFR cc = %r' % (sorted(cc1),))
        else:
            print('INFR cc1 = %r' % (sorted(cc1),))
            print('INFR cc2 = %r' % (sorted(cc2),))

        # if (nid1 == nid2) != (nid1_ == nid2_):
        #     util.cprint('DISAGREEMENT IN GRAPH AND DB', 'red')
        # else:
        #     util.cprint('GRAPH AND DB AGREE', 'green')

        # print('IBS  NAMES: nid1, nid2 = %r, %r' % (nid1_, nid2_))
        # if nid1_ == nid2_:
        #     print('IBS CC: %r' % (sorted(ibs.get_name_aids(nid1_)),))
        # else:
        #     print('IBS CC1: %r' % (sorted(ibs.get_name_aids(nid1_)),))
        #     print('IBS CC2: %r' % (sorted(ibs.get_name_aids(nid2_)),))

        # Does this exist in annotmatch?
        # in_am = ibs.get_annotmatch_rowid_from_undirected_superkey([aid1], [aid2])
        # print('in_am = %r' % (in_am,))

        # Does this exist in staging?
        # staging_rowids = ibs.get_review_rowids_from_edges([(aid1, aid2)])[0]
        # print('staging_rowids = %r' % (staging_rowids,))

        # if False:
        #     # Make absolutely sure
        #     stagedf = ibs.staging.get_table_as_pandas('reviews')
        #     aid_cols = ['annot_1_rowid', 'annot_2_rowid']
        #     has_aid1 = (stagedf[aid_cols] == aid1).any(axis=1)
        #     from_aid1 = stagedf[has_aid1]
        #     conn_aid2 = (from_aid1[aid_cols] == aid2).any(axis=1)
        #     print('# connections = %r' % (conn_aid2.sum(),))

        # Next check indirect relationships
        graph = infr.graph
        if cc1 != cc2:
            edge_df1 = get_aug_df(nxu.edges_between(graph, cc1))
            edge_df2 = get_aug_df(nxu.edges_between(graph, cc2))
            print_df(edge_df1, 'Inside1')

            print_df(edge_df2, 'Inside1')

            out_df1 = get_aug_df(nxu.edges_outgoing(graph, cc1))
            print_df(out_df1, 'Outgoing1')

            out_df2 = get_aug_df(nxu.edges_outgoing(graph, cc2))
            print_df(out_df2, 'Outgoing2')
        else:
            subgraph = infr.pos_graph.subgraph(cc1)
            print('Shortest path between endpoints')
            print(nx.shortest_path(subgraph, aid1, aid2))

        edge_df3 = get_aug_df(nxu.edges_between(graph, cc1, cc2))
        print_df(edge_df3, 'Between')

    def node_tag_hist(infr):
        tags_list = infr.ibs.get_annot_case_tags(infr.aids)
        tag_hist = util.tag_hist(tags_list)
        return tag_hist

    def edge_tag_hist(infr):
        tags_list = list(infr.gen_edge_values('tags', None))
        tag_hist = util.tag_hist(tags_list)
        return tag_hist

    def match_state_df(infr, index):
        """
        Returns the current matching state of a list of edges.

        PERHAPS WE SHOULD DEPRICATE THIS FUNCTION?

        Note:
            This does NOT use the IBEIS database state, where as the original
            version of this function did.

        CommandLine:
            python -m graphid.core.mixin_helpers Convenience.match_state_df

        Example:
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(num_pccs=2, p_incomp=.8, size=4)
            >>> index = list(infr.edges())
            >>> print(infr.match_state_df(index))
                       NEGTV  POSTV  INCMP
            aid1 aid2
            1    3     False  False   True
                 4     False  False   True
                 2     False   True  False
            2    3     False  False   True
                 4     False  False   True
            3    4     False   True  False
                 5     False  False   True
            5    8     False  False   True
                 7     False  False   True
                 6     False  False   True
            6    8     False  False   True
                 7     False  False   True
            7    8     False  False   True
        """
        index = util.ensure_multi_index(index, ('aid1', 'aid2'))
        aid_pairs = np.asarray(index.tolist())
        aid_pairs = aid_pairs.reshape(-1, 2)
        # is_same = np.array(
        #     [infr.pos_graph.are_nodes_connected(u, v) for u, v in aid_pairs])
        u_nids = np.array(list(infr.gen_node_values('name_label', [
            u for u, v in aid_pairs])))
        v_nids = np.array(list(infr.gen_node_values('name_label', [
            v for u, v in aid_pairs])))
        is_same = np.equal(u_nids, v_nids)

        edge_states = infr.gen_edge_values('evidence_decision', edges=aid_pairs,
                                           default=UNREV, on_missing='default')
        is_comp = np.array([s == INCMP for s in edge_states])

        if hasattr(pd.DataFrame, 'from_items'):
            match_state_df = pd.DataFrame.from_items([
                (NEGTV, ~is_same & is_comp),
                (POSTV,  is_same & is_comp),
                (INCMP, ~is_comp),
            ])
        else:
            match_state_df = pd.DataFrame(ub.odict([
                (NEGTV, ~is_same & is_comp),
                (POSTV,  is_same & is_comp),
                (INCMP, ~is_comp),
            ]))
        match_state_df.index = index
        return match_state_df


class DummyEdges:

    def ensure_mst(infr, label='name_label', meta_decision=SAME):
        """
        Ensures that all names are names are connected.

        Args:
            label (str): node attribute to use as the group id to form the mst.
            meta_decision (str): if specified adds clique edges as feedback
                items with this decision. Otherwise the edges are only
                explicitly added to the graph.  This makes feedback items with
                user_id=algo:mst and with a confidence of guessing.

        Example:
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(num_pccs=3, size=4)
            >>> assert infr.status()['nCCs'] == 3
            >>> infr.clear_edges()
            >>> assert infr.status()['nCCs'] == 12
            >>> infr.ensure_mst()
            >>> assert infr.status()['nCCs'] == 3
        """
        infr.print('ensure_mst', 1)
        new_edges = infr.find_mst_edges(label=label)
        # Add new MST edges to original graph
        infr.print('adding %d MST edges' % (len(new_edges)), 2)
        infr.add_feedback_from(new_edges, meta_decision=SAME,
                               confidence=const.CONFIDENCE.CODE.GUESSING,
                               user_id='algo:mst', verbose=False)

    def ensure_cliques(infr, label='name_label', meta_decision=None):
        """
        Force each name label to be a clique.

        Args:
            label (str): node attribute to use as the group id to form the
                cliques.
            meta_decision (str): if specified adds clique edges as feedback
                items with this decision. Otherwise the edges are only
                explicitly added to the graph.

        Args:
            label (str): defaults to 'name_label'
            meta_decision (str): if specified, the feedback edges added are
                added this meta decision and with the `user_id=algo:clique`.

        CommandLine:
            python -m graphid.core.mixin_helpers ensure_cliques

        Example:
            >>> from graphid import demo
            >>> label = 'name_label'
            >>> infr = demo.demodata_infr(num_pccs=3, size=5)
            >>> print(ub.urepr(infr.status()))
            >>> assert infr.status()['nEdges'] < 33
            >>> infr.ensure_cliques()
            >>> print(ub.urepr(infr.status()))
            >>> assert infr.status()['nEdges'] == 31
            >>> assert infr.status()['nUnrevEdges'] == 12
            >>> assert len(list(infr.find_clique_edges(label))) > 0
            >>> infr.ensure_cliques(meta_decision=SAME)
            >>> assert infr.status()['nUnrevEdges'] == 0
            >>> assert len(list(infr.find_clique_edges(label))) == 0
        """
        infr.print('ensure_cliques', 1)
        new_edges = infr.find_clique_edges(label)
        infr.print('ensuring %d clique edges' % (len(new_edges)), 2)
        if meta_decision is None:
            infr.ensure_edges_from(new_edges)
        else:
            infr.add_feedback_from(new_edges, meta_decision=SAME,
                                   confidence=const.CONFIDENCE.CODE.GUESSING,
                                   user_id='algo:clique', verbose=False)
        # infr.assert_disjoint_invariant()

    def ensure_full(infr):
        """
        Explicitly places all edges, but does not make any feedback items
        """
        infr.print('ensure_full with %d nodes' % (len(infr.graph)), 2)
        new_edges = list(nx.complement(infr.graph).edges())
        infr.ensure_edges_from(new_edges)

    def find_clique_edges(infr, label='name_label'):
        """
        Augmenting edges that would complete each the specified cliques.
        (based on the group inferred from `label`)

        Args:
            label (str): node attribute to use as the group id to form the
                cliques.
        """
        node_to_label = infr.get_node_attrs(label)
        label_to_nodes = ub.group_items(node_to_label.keys(),
                                        node_to_label.values())
        new_edges = []
        for label, nodes in label_to_nodes.items():
            for edge in it.combinations(nodes, 2):
                if infr.edge_decision(edge) == UNREV:
                    new_edges.append(edge)
        return new_edges

    def find_mst_edges(infr, label='name_label'):
        """
        Returns edges to augment existing PCCs (by label) in order to ensure
        they are connected with positive edges.

        Example:
            >>> # DISABLE_DOCTEST
            >>> from graphid.core.mixin_helpers import *  # NOQA
            >>> import ibeis
            >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
            >>> infr = ibeis.AnnotInference(ibs, 'all', autoinit=True)
            >>> label = 'orig_name_label'
            >>> label = 'name_label'
            >>> infr.find_mst_edges()
            >>> infr.ensure_mst()

        Ignore:
            old_mst_edges = [
                e for e, d in infr.edges(data=True)
                if d.get('user_id', None) == 'algo:mst'
            ]
            infr.graph.remove_edges_from(old_mst_edges)
            infr.pos_graph.remove_edges_from(old_mst_edges)
            infr.neg_graph.remove_edges_from(old_mst_edges)
            infr.incomp_graph.remove_edges_from(old_mst_edges)

        """
        # Find clusters by labels
        node_to_label = infr.get_node_attrs(label)
        label_to_nodes = ub.group_items(node_to_label.keys(),
                                        node_to_label.values())

        weight_heuristic = False
        # infr.ibs is not None
        if weight_heuristic:
            annots = infr.ibs.annots(infr.aids)
            node_to_time = ub.dzip(annots, annots.time)
            node_to_view = ub.dzip(annots, annots.viewpoint_code)
            enabled_heuristics = {
                'view_weight',
                'time_weight',
            }

        def _heuristic_weighting(nodes, avail_uv):
            avail_uv = np.array(avail_uv)
            weights = np.ones(len(avail_uv))

            if 'view_weight' in enabled_heuristics:
                from graphid.core import _rhomb_dist
                view_edge = [(node_to_view[u], node_to_view[v])
                             for (u, v) in avail_uv]
                view_weight = np.array([
                    _rhomb_dist.VIEW_CODE_DIST[(v1, v2)]
                    for (v1, v2) in view_edge
                ])
                # Assume comparable by default and prefer undefined
                # more than probably not, but less than definately so.
                view_weight[np.isnan(view_weight)] = 1.5
                # Prefer viewpoint 10x more than time
                weights += 10 * view_weight

            if 'time_weight' in enabled_heuristics:
                # Prefer linking annotations closer in time
                times = list(ub.take(node_to_time, nodes))
                maxtime = util.safe_max(times, fill=1, nans=False)
                mintime = util.safe_min(times, fill=0, nans=False)
                time_denom = maxtime - mintime
                # Try linking by time for lynx data
                time_delta = np.array([
                    abs(node_to_time[u] - node_to_time[v])
                    for u, v in avail_uv
                ])
                time_weight = time_delta / time_denom
                weights += time_weight

            weights = np.array(weights)
            weights[np.isnan(weights)] = 1.0

            avail = [(u, v, {'weight': w})
                     for (u, v), w in zip(avail_uv, weights)]
            return avail

        new_edges = []
        prog = ub.ProgIter(list(label_to_nodes.keys()),
                           desc='finding mst edges',
                           enabled=infr.verbose > 0)
        for nid in prog:
            nodes = set(label_to_nodes[nid])
            if len(nodes) == 1:
                continue
            # We want to make this CC connected
            pos_sub = infr.pos_graph.subgraph(nodes, dynamic=False)
            impossible = set(it.starmap(e_, it.chain(
                nxu.edges_inside(infr.neg_graph, nodes),
                nxu.edges_inside(infr.incomp_graph, nodes),
                # nxu.edges_inside(infr.unknown_graph, nodes),
            )))
            if len(impossible) == 0 and not weight_heuristic:
                # Simple mst augmentation
                aug_edges = list(nxu.k_edge_augmentation(pos_sub, k=1))
            else:
                complement = it.starmap(e_, nxu.complement_edges(pos_sub))
                avail_uv = [
                    (u, v) for u, v in complement if (u, v) not in impossible
                ]
                if weight_heuristic:
                    # Can do heuristic weighting to improve the MST
                    avail = _heuristic_weighting(nodes, avail_uv)
                else:
                    avail = avail_uv
                # print(len(pos_sub))
                try:
                    aug_edges = list(nxu.k_edge_augmentation(
                        pos_sub, k=1, avail=avail))
                except nx.NetworkXUnfeasible:
                    print('Warning: MST augmentation is not feasible')
                    print('explicit negative edges might disconnect a PCC')
                    aug_edges = list(nxu.k_edge_augmentation(
                        pos_sub, k=1, avail=avail, partial=True))
            new_edges.extend(aug_edges)
        prog.ensure_newline()

        for edge in new_edges:
            assert not infr.graph.has_edge(*edge), (
                'alrady have edge={}'.format(edge))
        return new_edges

    def find_connecting_edges(infr):
        """
        Searches for a small set of edges, which if reviewed as positive would
        ensure that each PCC is k-connected.  Note that in somes cases this is
        not possible
        """
        label = 'name_label'
        node_to_label = infr.get_node_attrs(label)
        label_to_nodes = ub.group_items(node_to_label.keys(),
                                        node_to_label.values())

        # k = infr.params['redun.pos']
        k = 1
        new_edges = []
        prog = ub.ProgIter(list(label_to_nodes.keys()),
                           desc='finding connecting edges',
                           enabled=infr.verbose > 0)
        for nid in prog:
            nodes = set(label_to_nodes[nid])
            G = infr.pos_graph.subgraph(nodes, dynamic=False)
            impossible = nxu.edges_inside(infr.neg_graph, nodes)
            impossible |= nxu.edges_inside(infr.incomp_graph, nodes)

            candidates = set(nx.complement(G).edges())
            candidates.difference_update(impossible)

            aug_edges = nxu.k_edge_augmentation(G, k=k, avail=candidates)
            new_edges += aug_edges
        prog.ensure_newline()
        return new_edges

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/graphid/graphid.core/mixin_helpers.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
