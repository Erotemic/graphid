import ubelt as ub
import numpy as np
import pandas as pd
from graphid import util
from graphid.core.state import POSTV, NEGTV, INCMP


class Groundtruth:

    def is_comparable(infr, aid_pairs, allow_guess=True):
        """
        Guesses by default when real comparable information is not available.
        """
        if infr.ibs is not None:
            return infr.ibeis_is_comparable(aid_pairs, allow_guess)
        is_comp = list(infr.gen_edge_values('gt_comparable', edges=aid_pairs,
                                            default=True,
                                            on_missing='default'))
        return np.array(is_comp)

    def is_photobomb(infr, aid_pairs):
        if infr.ibs is not None:
            return infr.ibeis_is_photobomb(aid_pairs)
        return np.array([False] * len(aid_pairs))

    def is_same(infr, aid_pairs):
        if infr.ibs is not None:
            return infr.ibeis_is_same(aid_pairs)
        node_dict = infr.graph.nodes
        nid1 = [node_dict[n1]['orig_name_label']
                for n1, n2 in aid_pairs]
        nid2 = [node_dict[n2]['orig_name_label']
                for n1, n2 in aid_pairs]
        return np.equal(nid1, nid2)

    def apply_edge_truth(infr, edges=None):
        if edges is None:
            edges = list(infr.edges())
        edge_truth_df = infr.match_state_df(edges)
        edge_truth = edge_truth_df.idxmax(axis=1).to_dict()
        infr.set_edge_attrs('truth', edge_truth)
        infr.edge_truth.update(edge_truth)

    def match_state_df(infr, index):
        """ Returns groundtruth state based on ibeis controller """
        index = util.ensure_multi_index(index, ('aid1', 'aid2'))
        aid_pairs = np.asarray(index.tolist())
        aid_pairs = aid_pairs.reshape(-1, 2)
        is_same = infr.is_same(aid_pairs)
        is_comp = infr.is_comparable(aid_pairs)
        match_state_df = pd.DataFrame.from_items([
            (NEGTV, ~is_same & is_comp),
            (POSTV,  is_same & is_comp),
            (INCMP, ~is_comp),
        ])
        match_state_df.index = index
        return match_state_df

    def match_state_gt(infr, edge):
        if edge in infr.edge_truth:
            truth = infr.edge_truth[edge]
        elif hasattr(infr, 'dummy_verif'):
            truth = infr.dummy_verif._get_truth(edge)
        else:
            aid_pairs = np.asarray([edge])
            is_same = infr.is_same(aid_pairs)[0]
            is_comp = infr.is_comparable(aid_pairs)[0]
            match_state = pd.Series(dict([
                (NEGTV, ~is_same & is_comp),
                (POSTV,  is_same & is_comp),
                (INCMP, ~is_comp),
            ]))
            truth = match_state.idxmax()
        return truth

    def edge_attr_df(infr, key, edges=None, default=ub.NoParam):
        """ constructs DataFrame using current predictions """
        edge_states = infr.gen_edge_attrs(key, edges=edges, default=default)
        edge_states = list(edge_states)
        if isinstance(edges, pd.MultiIndex):
            index = edges
        else:
            if edges is None:
                edges_ = util.take_column(edge_states, 0)
            else:
                edges_ = list(map(tuple, util.aslist(edges)))
            index = pd.MultiIndex.from_tuples(edges_, names=('aid1', 'aid2'))
        records = util.itake_column(edge_states, 1)
        edge_df = pd.Series.from_array(records)
        edge_df.name = key
        edge_df.index = index
        return edge_df
