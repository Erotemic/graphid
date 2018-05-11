# -*- coding: utf-8 -*-
import itertools as it
import numpy as np
import pandas as pd
import ubelt as ub
from graphid.core.state import POSTV, NEGTV, INCMP, UNREV  # NOQA
from graphid.core.state import SAME, DIFF, NULL  # NOQA
from graphid import util


class DummyVerif(object):
    """
    generates dummy scores between edges (not necesarilly in the graph)

    CommandLine:
        python -m graphid.demo DummyVerif:1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from graphid.demo import *  # NOQA
        >>> from graphid.core import demo
        >>> import networkx as nx
        >>> kwargs = dict(num_pccs=6, p_incon=.5, size_std=2)
        >>> infr = demo.demodata_infr(**kwargs)
        >>> infr.dummy_verif.predict_edges([(1, 2)])
        >>> infr.dummy_verif.predict_edges([(1, 21)])
        >>> assert len(infr.dummy_verif.infr.task_probs['match_state']) == 2
    """
    def __init__(verif, infr):
        verif.rng = np.random.RandomState(4033913)
        verif.dummy_params = {
            NEGTV: {'mean': .2, 'std': .25},
            POSTV: {'mean': .85, 'std': .2},
            INCMP: {'mean': .15, 'std': .1},
        }
        verif.score_dist = util.randn

        verif.infr = infr
        verif.orig_nodes = set(infr.aids)
        verif.orig_labels = infr.get_node_attrs('orig_name_label')
        verif.orig_groups = ub.invert_dict(verif.orig_labels, False)
        verif.orig_groups = ub.map_vals(set, verif.orig_groups)

    def show_score_probs(verif):
        """
        CommandLine:
            python -m graphid.demo DummyVerif.show_score_probs --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from graphid.demo import *  # NOQA
            >>> infr = AnnotInference(None)
            >>> verif = DummyVerif(infr)
            >>> verif.show_score_probs()
            >>> util.show_if_requested()
        """
        import plottool as pt
        dist = verif.score_dist
        n = 100000
        for key in verif.dummy_params.keys():
            probs = dist(shape=[n], rng=verif.rng, a_max=1, a_min=0,
                          **verif.dummy_params[key])
            color = verif.infr._get_truth_colors()[key]
            pt.plt.hist(probs, bins=100, label=key, alpha=.8, color=color)
        pt.legend()

    def dummy_ranker(verif, u, K=10):
        """
        simulates the ranking algorithm. Order is defined using the dummy vsone
        scores, but tests are only applied to randomly selected gt and gf
        pairs. So, you usually will get a gt result, but you might not if all
        the scores are bad.
        """
        infr = verif.infr

        nid = verif.orig_labels[u]
        others = verif.orig_groups[nid]
        others_gt = sorted(others - {u})
        others_gf = sorted(verif.orig_nodes - others)

        # rng = np.random.RandomState(u + 4110499444 + len(others))
        rng = verif.rng

        vs_list = []
        k_gt = min(len(others_gt), max(1, K // 2))
        k_gf = min(len(others_gf), max(1, K * 4))
        if k_gt > 0:
            gt = rng.choice(others_gt, k_gt, replace=False)
            vs_list.append(gt)
        if k_gf > 0:
            gf = rng.choice(others_gf, k_gf, replace=False)
            vs_list.append(gf)

        u_edges = [infr.e_(u, v) for v in it.chain.from_iterable(vs_list)]
        u_probs = np.array(infr.dummy_verif.predict_edges(u_edges))
        # infr.set_edge_attrs('prob_match', ub.dzip(u_edges, u_probs))

        # Need to determenistically sort here
        # sortx = np.argsort(u_probs)[::-1][0:K]

        sortx = np.argsort(u_probs)[::-1][0:K]
        ranked_edges = list(ub.take(u_edges, sortx))
        # assert len(ranked_edges) == K
        return ranked_edges

    def find_candidate_edges(verif, K=10):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from graphid import demo
            >>> import networkx as nx
            >>> kwargs = dict(num_pccs=40, size=2)
            >>> infr = demo.demodata_infr(**kwargs)
            >>> edges = list(infr.dummy_verif.find_candidate_edges(K=100))
            >>> scores = np.array(infr.dummy_verif.predict_edges(edges))
        """
        new_edges = []
        nodes = list(verif.infr.graph.nodes())
        for u in nodes:
            new_edges.extend(verif.dummy_ranker(u, K=K))
        # print('new_edges = %r' % (ub.hash_data(new_edges),))
        new_edges = set(new_edges)
        return new_edges

    def _get_truth(verif, edge):
        infr = verif.infr
        if edge in infr.edge_truth:
            return infr.edge_truth[edge]
        node_dict = infr.graph.nodes
        nid1 = node_dict[edge[0]]['orig_name_label']
        nid2 = node_dict[edge[1]]['orig_name_label']
        return POSTV if nid1 == nid2 else NEGTV

    def predict_proba_df(verif, edges):
        """
        CommandLine:
            python -m graphid.demo DummyVerif.predict_edges

        Example:
            >>> # ENABLE_DOCTEST
            >>> from graphid import demo
            >>> import networkx as nx
            >>> kwargs = dict(num_pccs=40, size=2)
            >>> infr = demo.demodata_infr(**kwargs)
            >>> verif = infr.dummy_verif
            >>> edges = list(infr.graph.edges())
            >>> probs = verif.predict_proba_df(edges)
            >>> #print('scores = %r' % (scores,))
            >>> #hashid = ub.hash_data(scores)
            >>> #print('hashid = %r' % (hashid,))
            >>> #assert hashid == 'cdlkytilfeqgmtsihvhqwffmhczqmpil'
        """
        infr = verif.infr
        edges = list(it.starmap(verif.infr.e_, edges))
        prob_cache = infr.task_probs['match_state']
        is_miss = np.array([e not in prob_cache for e in edges])
        # is_hit = ~is_miss
        if np.any(is_miss):
            miss_edges = list(ub.compress(edges, is_miss))
            miss_truths = [verif._get_truth(edge) for edge in miss_edges]
            grouped_edges = ub.group_items(miss_edges, miss_truths,
                                           sorted_=False)
            # Need to make this determenistic too
            states = [POSTV, NEGTV, INCMP]
            for key in sorted(grouped_edges.keys()):
                group = grouped_edges[key]
                probs0 = util.randn(shape=[len(group)], rng=verif.rng, a_max=1,
                                    a_min=0, **verif.dummy_params[key])
                # Just randomly assign other probs
                probs1 = verif.rng.rand(len(group)) * (1 - probs0)
                probs2 = 1 - (probs0 + probs1)
                for edge, probs in zip(group, zip(probs0, probs1, probs2)):
                    prob_cache[edge] = ub.dzip(states, probs)

        probs = pd.DataFrame(
            list(ub.take(prob_cache, edges)),
            index=util.ensure_multi_index(edges, ('aid1', 'aid2'))
        )
        return probs

    def predict_edges(verif, edges):
        pos_scores = verif.predict_proba_df(edges)[POSTV]
        return pos_scores
