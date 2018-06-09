import numpy as np
import ubelt as ub
from graphid.core.state import (POSTV, NEGTV, INCMP, NULL)  # NOQA
from graphid.core import abstract


class InfrCallbacks(object):
    """
    Methods relating to callbacks that must be registered with the inference
    object for it to work properly.
    """

    def set_ranker(infr, ranker):
        """
        ranker should be a function that accepts a list of annotation ids and
        return a list of the top K ranked annotations.
        """
        if not isinstance(ranker, abstract.Ranker):
            for key in abstract.Ranker.__dict__:
                if not key.startswith('_'):
                    if not hasattr(ranker, key):
                        raise ValueError('ranker must implement {}'.format(key))
        infr.ranker = ranker

    def set_verifier(infr, verifier, task='match_state'):
        """
        verifier should be a function that accepts a list of annotation pairs
        and produces the 3-state match_state probabilities.
        """
        if not isinstance(verifier, abstract.Verifier):
            for key in abstract.Verifier.__dict__:
                if not key.startswith('_'):
                    if not hasattr(verifier, key):
                        raise ValueError('verifier must implement {}'.format(key))
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
        Uses the registered ranking algorithm to produce new candidaate edges.
        These are then scored by the verification algorithm and inserted into
        the priority queue.

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
            infr.print('dummy vsone params =' + ub.repr2(
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


class InfrCandidates(object):
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
            raise AssertionError('Verifiers are needed to predict probabilities')

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

    def ensure_priority_scores(infr, edges):
        """
        Ensures that priority attributes are assigned to the edges.
        This does not change the state of the queue.

        Args:
            edges (List[Tuple[int, int]]): edges to ensure have priority scores

        Doctest:
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(num_pccs=6, p_incon=.5, size_std=2)
            >>> edges = list(infr.edges())
            >>> infr.ensure_priority_scores(edges)
        """
        if infr.verifiers:
            infr.print('Prioritizing {} edges with one-vs-one probs'.format(
                    len(edges)), 1)
            metric = 'dynamic_priority'
            priority = list(infr.gen_dynamic_priority(edges))
        elif infr.cm_list is not None:
            infr.print(
                'Prioritizing {} edges with one-vs-vsmany scores'.format(
                    len(edges), 1))
            # Not given any deploy classifier, this is the best we can do
            scores = infr._make_lnbnn_scores(edges)
            metric = 'normscore'
            priority = scores
        else:
            infr.print(
                'WARNING: No verifiers to prioritize {} edge(s)'.format(
                    len(edges)))
            metric = 'random'
            priority = np.zeros(len(edges)) + 1e-6

        infr.set_edge_attrs(metric, ub.dzip(edges, priority))
        return metric, priority

    def gen_dynamic_priority(infr, edges):
        """
        Generates the dynamic priority of each edge. This is the positive
        probability if the edge is between two PCCs and the negative if it is
        within the same PCC.

        If verifiers are not set, this just returns random numbers.

        Notes:
            assumes a verifier exists to populate infr.task_probs['match_state']

        Args:
            edges (list): edges of interest

        Yields:
            float: priority score
        """
        if not infr.verifiers:
            raise AssertionError('must have verifiers')
            for edge in edges:
                aid1, aid2 = edge
                score = 0
                # Hack in a measure where we prefer edges with a lower degree.
                d1 = infr.graph.degree[aid1] - infr.unreviewed_graph.degree[aid1]
                d2 = infr.graph.degree[aid2] - infr.unreviewed_graph.degree[aid2]
                score += (5 - min((d1 + d2) / 2, 5)) / 10.0
                yield score
        else:
            # raise AssertionError(
            #     'need a match_state verifier to use dynamic priorities')
            infr.ensure_task_probs(edges)

            prioritize_nonpos = (infr.params['autoreview.enabled'] and
                                 infr.params['autoreview.prioritize_nonpos'])

            all_match_probs = infr.task_probs['match_state']
            match_thresh = infr.task_thresh['match_state']

            for edge in edges:
                match_probs = all_match_probs[edge]
                # If edges are between different PCCs, prioritize by POSTV probs
                # If edges are within the same PCC, prioritize by NEGTV probs
                # These edges are the most likely to cause splits and merges
                aid1, aid2 = edge
                nid1, nid2 = infr.pos_graph.node_labels(aid1, aid2)
                score = match_probs[NEGTV] if nid1 == nid2 else match_probs[POSTV]

                if prioritize_nonpos:
                    if match_thresh[POSTV] > match_probs[POSTV]:
                        score = max(score, match_probs[POSTV]) + 1
                    if match_thresh[NEGTV] > match_probs[NEGTV]:
                        score = max(score, match_probs[NEGTV]) + 1
                    if match_thresh[INCMP] > match_probs[INCMP]:
                        score = max(score, match_probs[NEGTV]) + 1

                # Hack in a measure where we prefer edges with a lower degree.
                d1 = infr.graph.degree[aid1] - infr.unreviewed_graph.degree[aid1]
                d2 = infr.graph.degree[aid2] - infr.unreviewed_graph.degree[aid2]
                score += (5 - min((d1 + d2) / 2, 5)) / 10.0
                yield score


if __name__ == '__main__':
    """
    CommandLine:
        python -m graphid.core.mixin_callbacks all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
