import numpy as np
import ubelt as ub
import pandas as pd
from graphid.core.state import (POSTV, NEGTV, INCMP, NULL)  # NOQA
from graphid import util


class InfrCallbacks:
    """
    Methods relating to callbacks that must be registered with the inference
    object for it to work properly.
    """

    def set_ranker(infr, ranker):
        """
        ranker should be a function that accepts a list of annotation ids and
        return a list of the top K ranked annotations.
        """
        infr.ranker = ranker

    def set_verifier(infr, verifier, task='match_state'):
        """
        verifier should be a function that accepts a list of annotation pairs
        and produces the 3-state match_state probabilities.
        """
        if infr.verifiers is None:
            infr.verifiers = {}
        infr.verifiers[task] = verifier
        infr.verifier = verifier

    # def set_edge_attr_predictor(infr, func):
    #     infr.predict_edge_attrs = func

    # def set_node_attr_predictor(infr, func):
    #     infr.predict_node_attrs = func

    # def _default_candidate_edge_search(infr):
    #     raise NotImplementedError

    def refresh_candidate_edges(infr):
        """
        CommandLine:
            python -m graphid.core.mixin_callbacks InfrCallbacks.refresh_candidate_edges

        Example:
            >>> from graphid import demo
            >>> kwargs = dict(num_pccs=40, size=2)
            >>> infr = demo.demodata_infr(**kwargs)
            >>> infr.refresh_candidate_edges()
        """
        infr.print('refresh_candidate_edges', 1)
        infr.assert_consistency_invariant()

        if hasattr(infr, 'dummy_verif'):
            infr.print('Searching for dummy candidates')
            infr.print('dummy vsone params =' + ub.urepr(
                infr.dummy_verif.dummy_params, nl=1, si=True))

        if infr.ranker is None:
            raise Exception(
                'No method available to search for candidate edges')

        ranks_top = infr.params['ranking.ntop']
        qaids = list(infr.aids)
        rankings = infr.ranker.predict_rankings(qaids, K=ranks_top)
        candidate_edges = [
            infr.e_(aid, v)
            for aid, rankings in zip(qaids, rankings)
            for v in rankings
        ]
        infr.add_candidate_edges(candidate_edges)
        infr.assert_consistency_invariant()


class InfrCandidates:
    """
    Methods that should be used by callbacks to add new edges to be considered
    as candidates in the priority queue.
    """

    def add_candidate_edges(infr, candidate_edges):
        candidate_edges = list(candidate_edges)
        new_edges = infr.ensure_edges_from(candidate_edges)

        if infr.params['redun.enabled']:
            priority_edges = list(infr.filter_edges_flagged_as_redun(
                candidate_edges))
            infr.print('Got {} candidate edges, {} are new, '
                       'and {} are non-redundant'.format(
                           len(candidate_edges), len(new_edges),
                           len(priority_edges)))
        else:
            infr.print('Got {} candidate edges and {} are new'.format(
                len(candidate_edges), len(new_edges)))
            priority_edges = candidate_edges

        if len(priority_edges) > 0:
            priority_edges = list(priority_edges)
            metric, priority = infr.ensure_priority_scores(priority_edges)
            infr.prioritize(metric=metric, edges=priority_edges, scores=priority)
            if hasattr(infr, 'on_new_candidate_edges'):
                # hack callback for demo
                infr.on_new_candidate_edges(infr, new_edges)
        return len(priority_edges)

    def ensure_task_probs(infr, edges):
        """
        Ensures that probabilities are assigned to the edges.
        This gaurentees that infr.task_probs contains data for edges.
        (Currently only the primary task is actually ensured)

        CommandLine:
            python -m graphid.core.mixin_callbacks InfrCandidates.ensure_task_probs

        Doctest:
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(num_pccs=6, p_incon=.5, size_std=2)
            >>> edges = list(infr.edges())
            >>> infr.ensure_task_probs(edges)
            >>> assert all([np.isclose(sum(p.values()), 1)
            >>>             for p in infr.task_probs['match_state'].values()])
        """
        if not infr.verifiers:
            raise Exception('Verifiers are needed to predict probabilities')

        # Construct pairwise features on edges in infr
        primary_task = 'match_state'

        match_task = infr.task_probs[primary_task]
        need_flags = [e not in match_task for e in edges]

        if any(need_flags):
            need_edges = list(ub.compress(edges, need_flags))
            infr.print('There are {} edges without probabilities'.format(
                    len(need_edges)), 1)

            # Only recompute for the needed edges
            # task_probs = infr._make_task_probs(need_edges)
            task_probs = {
                primary_task: infr.verifier.predict_proba_df(need_edges)
            }
            # Store task probs in internal data structure
            # FIXME: this is slow
            for task, probs in task_probs.items():
                probs_dict = probs.to_dict(orient='index')
                if task not in infr.task_probs:
                    infr.task_probs[task] = probs_dict
                else:
                    infr.task_probs[task].update(probs_dict)

                # Set edge task attribute as well
                infr.set_edge_attrs(task, probs_dict)

    def ensure_priority_scores(infr, priority_edges):
        """
        Ensures that priority attributes are assigned to the edges.
        This does not change the state of the queue.

        Doctest:
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(num_pccs=6, p_incon=.5, size_std=2)
            >>> edges = list(infr.edges())
            >>> infr.ensure_priority_scores(edges)
        """
        if infr.verifiers:
            infr.print('Prioritizing {} edges with one-vs-one probs'.format(
                    len(priority_edges)), 1)

            infr.ensure_task_probs(priority_edges)

            primary_task = 'match_state'
            match_probs = infr.task_probs[primary_task]
            primary_thresh = infr.task_thresh[primary_task]

            # Read match_probs into a DataFrame
            primary_probs = pd.DataFrame(
                list(ub.take(match_probs, priority_edges)),
                index=util.ensure_multi_index(priority_edges, ('aid1', 'aid2'))
            )

            # Convert match-state probabilities into priorities
            prob_match = primary_probs[POSTV]

            # Initialize priorities to probability of matching
            default_priority = prob_match.copy()

            # If the edges are currently between the same individual, then
            # prioritize by non-positive probability (because those edges might
            # expose an inconsistency)
            already_pos = [
                infr.pos_graph.node_label(u) == infr.pos_graph.node_label(v)
                for u, v in priority_edges
            ]
            default_priority[already_pos] = 1 - default_priority[already_pos]

            if infr.params['autoreview.enabled']:
                if infr.params['autoreview.prioritize_nonpos']:
                    # Give positives that pass automatic thresholds high priority
                    _probs = primary_probs[POSTV]
                    flags = _probs > primary_thresh[POSTV]
                    default_priority[flags] = np.maximum(default_priority[flags],
                                                         _probs[flags]) + 1

                    # Give negatives that pass automatic thresholds high priority
                    _probs = primary_probs[NEGTV]
                    flags = _probs > primary_thresh[NEGTV]
                    default_priority[flags] = np.maximum(default_priority[flags],
                                                         _probs[flags]) + 1

                    # Give not-comps that pass automatic thresholds high priority
                    _probs = primary_probs[INCMP]
                    flags = _probs > primary_thresh[INCMP]
                    default_priority[flags] = np.maximum(default_priority[flags],
                                                         _probs[flags]) + 1

            infr.set_edge_attrs('prob_match', prob_match.to_dict())
            infr.set_edge_attrs('default_priority', default_priority.to_dict())

            metric = 'default_priority'
            priority = default_priority
        elif infr.cm_list is not None:
            infr.print(
                'Prioritizing {} edges with one-vs-vsmany scores'.format(
                    len(priority_edges)))
            # Not given any deploy classifier, this is the best we can do
            scores = infr._make_lnbnn_scores(priority_edges)
            metric = 'normscore'
            priority = scores
        else:
            infr.print(
                'WARNING: No verifiers to prioritize {} edge(s)'.format(
                    len(priority_edges)))
            metric = 'random'
            priority = np.zeros(len(priority_edges)) + 1e-6

        infr.set_edge_attrs(metric, ub.dzip(priority_edges, priority))
        return metric, priority


if __name__ == '__main__':
    """
    CommandLine:
        python -m graphid.core.mixin_callbacks all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
