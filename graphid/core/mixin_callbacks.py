class InfrCallbacks(object):
    """
    Methods relating to callbacks that must be registered with the inference
    object for it to work properly.
    """

    def set_candidate_predictor(infr, func):
        infr.predict_candidate_edges = func

    def set_edge_attr_predictor(infr, func):
        infr.predict_edge_attrs = func

    def set_node_attr_predictor(infr, func):
        infr.predict_node_attrs = func

    def _default_candidate_edge_search(infr):
        raise NotImplementedError

    def refresh_candidate_edges(infr):
        print('Warning: matching algorithm callbacks are not implemented yet')


class InfrCandidates(object):
    """
    Methods that should be used by callbacks to add new edges to be considered
    as candidates in the priority queue.
    """

    def ensure_prioritized(infr, priority_edges):
        priority_edges = list(priority_edges)
        metric, priority = infr.ensure_priority_scores(priority_edges)
        infr.prioritize(metric=metric, edges=priority_edges, scores=priority)

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
            infr.ensure_prioritized(priority_edges)
            if hasattr(infr, 'on_new_candidate_edges'):
                # hack callback for demo
                infr.on_new_candidate_edges(infr, new_edges)
        return len(priority_edges)

    def ensure_priority_scores(infr, priority_edges):
        """
        Ensures that priority attributes are assigned to the edges.
        This does not change the state of the queue.

        Doctest:
            >>> from graphid.core import demo
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
                index=nxu.ensure_multi_index(priority_edges, ('aid1', 'aid2'))
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
                    len(priority_edges), 1))
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
