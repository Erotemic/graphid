# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import ubelt as ub
from graphid import util
from graphid.util import nx_utils as nxu
from graphid.util.nx_utils import e_
from graphid.core.state import POSTV, NEGTV, INCMP, UNREV  # NOQA


class AnnotInfrMatching(object):
    """
    Methods for running matching algorithms
    """

    def exec_matching(infr, qaids=None, daids=None, prog_hook=None,
                      cfgdict=None, name_method='node'):
        """
        Loads chip matches into the inference structure
        Uses graph name labeling and ignores ibeis labeling
        """
        infr._make_rankings(qaids, daids, prog_hook, cfgdict, name_method)

    def _set_vsmany_info(infr, qreq_, cm_list):
        infr.vsmany_qreq_ = qreq_
        infr.vsmany_cm_list = cm_list
        infr.cm_list = cm_list
        infr.qreq_ = qreq_

    def _make_rankings(infr, qaids=None, daids=None, prog_hook=None,
                       cfgdict=None, name_method='node'):
        #from graphid.core import graph_iden

        # TODO: expose other ranking algos like SMK
        rank_algo = 'LNBNN'
        infr.print('Exec {} ranking algorithm'.format(rank_algo), 1)
        ibs = infr.ibs
        if qaids is None:
            qaids = infr.aids
        qaids = util.ensure_iterable(qaids)
        if daids is None:
            daids = infr.aids
        if cfgdict is None:
            cfgdict = {
                # 'can_match_samename': False,
                'can_match_samename': True,
                'can_match_sameimg': True,
                # 'augment_queryside_hack': True,
                'K': 3,
                'Knorm': 3,
                'prescore_method': 'csum',
                'score_method': 'csum'
            }
            cfgdict.update(infr.ranker_params)
        # hack for using current nids
        if name_method == 'node':
            aids = sorted(set(util.aslist(qaids) + util.aslist(daids)))
            custom_nid_lookup = infr.get_node_attrs('name_label', aids)
        elif name_method == 'edge':
            custom_nid_lookup = {
                aid: nid for nid, cc in infr.pos_graph._ccs.items() for aid in cc
            }
        elif name_method == 'ibeis':
            custom_nid_lookup = None
        else:
            raise KeyError('Unknown name_method={}'.format(name_method))

        qreq_ = ibs.new_query_request(qaids, daids, cfgdict=cfgdict,
                                      custom_nid_lookup=custom_nid_lookup,
                                      verbose=infr.verbose >= 2)

        # cacher = qreq_.get_big_cacher()
        # if not cacher.exists():
        #     pass
        #     # import sys
        #     # sys.exit(1)

        cm_list = qreq_.execute(prog_hook=prog_hook)
        infr._set_vsmany_info(qreq_, cm_list)

        edges = set(infr._cm_breaking(
            cm_list, review_cfg={'ranks_top': 5}))
        return edges
        # return cm_list

    def _make_matches_from(infr, edges, config=None, prog_hook=None):
        from ibeis.algo.verif import pairfeat
        if config is None:
            config = infr.verifier_params
        extr = pairfeat.PairwiseFeatureExtractor(infr.ibs, config=config)
        match_list = extr._exec_pairwise_match(edges, prog_hook=prog_hook)
        return match_list

    def exec_vsone_subset(infr, edges, prog_hook=None):
        """
        Args:
            prog_hook (None): (default = None)

        Example:
            >>> # ENABLE_DOCTEST
            >>> infr = testdata_infr('testdb1')
            >>> infr.ensure_full()
            >>> edges = [(1, 2), (2, 3)]
            >>> result = infr.exec_vsone_subset(edges)
            >>> print(result)
        """
        match_list = infr._make_matches_from(edges, prog_hook)

        # TODO: is this code necessary anymore?
        vsone_matches = {e_(u, v): match
                         for (u, v), match in zip(edges, match_list)}
        infr.vsone_matches.update(vsone_matches)
        edge_to_score = {e: match.fs.sum() for e, match in
                         vsone_matches.items()}
        infr.graph.add_edges_from(edge_to_score.keys())
        infr.set_edge_attrs('score', edge_to_score)
        return match_list

    def lookup_cm(infr, aid1, aid2):
        """
        Get chipmatch object associated with an edge if one exists.
        """
        if infr.cm_list is None:
            return None, aid1, aid2
        # TODO: keep chip matches in dictionary by default?
        aid2_idx = util.make_index_lookup(
            [cm.qaid for cm in infr.cm_list])
        switch_order = False

        if aid1 in aid2_idx:
            idx = aid2_idx[aid1]
            cm = infr.cm_list[idx]
            if aid2 not in cm.daid2_idx:
                switch_order = True
                # raise KeyError('switch order')
        else:
            switch_order = True

        if switch_order:
            # switch order
            aid1, aid2 = aid2, aid1
            idx = aid2_idx[aid1]
            cm = infr.cm_list[idx]
            if aid2 not in cm.daid2_idx:
                raise KeyError('No ChipMatch for edge (%r, %r)' % (aid1, aid2))
        return cm, aid1, aid2

    def apply_match_edges(infr, review_cfg={}):
        """
        Adds results from one-vs-many rankings as edges in the graph
        """
        if infr.cm_list is None:
            infr.print('apply_match_edges - matching has not been run!')
            return
        infr.print('apply_match_edges', 1)
        edges = infr._cm_breaking(review_cfg)
        # Create match-based graph structure
        infr.print('apply_match_edges adding %d edges' % len(edges), 1)
        infr.graph.add_edges_from(edges)
        infr.apply_match_scores()

    def _cm_breaking(infr, cm_list=None, review_cfg={}):
        """
            >>> review_cfg = {}
        """
        if cm_list is None:
            cm_list = infr.cm_list
        ranks_top = review_cfg.get('ranks_top', None)
        ranks_bot = review_cfg.get('ranks_bot', None)

        # Construct K-broken graph
        edges = []

        if ranks_bot is None:
            ranks_bot = 0

        for count, cm in enumerate(cm_list):
            score_list = cm.annot_score_list
            rank_list = ub.argsort(score_list)[::-1]
            sortx = ub.argsort(rank_list)

            top_sortx = sortx[:ranks_top]
            bot_sortx = sortx[len(sortx) - ranks_bot:]
            short_sortx = list(ub.unique(top_sortx + bot_sortx))

            daid_list = list(ub.take(cm.daid_list, short_sortx))
            for daid in daid_list:
                u, v = (cm.qaid, daid)
                if v < u:
                    u, v = v, u
                edges.append((u, v))
        return edges

    def _cm_training_pairs(infr, qreq_=None, cm_list=None,
                           top_gt=2, mid_gt=2, bot_gt=2, top_gf=2,
                           mid_gf=2, bot_gf=2, rand_gt=2, rand_gf=2, rng=None):
        """
        Constructs training data for a pairwise classifier

        Example:
            >>> # ENABLE_DOCTEST
            >>> infr = testdata_infr('PZ_MTEST')
            >>> infr.exec_matching(cfgdict={
            >>>     'can_match_samename': True,
            >>>     'K': 4,
            >>>     'Knorm': 1,
            >>>     'prescore_method': 'csum',
            >>>     'score_method': 'csum'
            >>> })
            >>> exec(ut.execstr_funckw(infr._cm_training_pairs))
            >>> rng = np.random.RandomState(42)
            >>> aid_pairs = np.array(infr._cm_training_pairs(rng=rng))
            >>> print(len(aid_pairs))
            >>> assert np.sum(aid_pairs.T[0] == aid_pairs.T[1]) == 0
        """
        if qreq_ is None:
            cm_list = infr.cm_list
            qreq_ = infr.qreq_
        ibs = infr.ibs
        aid_pairs = []
        dnids = qreq_.get_qreq_annot_nids(qreq_.daids)
        # dnids = qreq_.get_qreq_annot_nids(qreq_.daids)
        rng = util.ensure_rng(rng)
        for cm in ub.ProgIter(cm_list, desc='building pairs'):
            all_gt_aids = cm.get_top_gt_aids(ibs)
            all_gf_aids = cm.get_top_gf_aids(ibs)
            gt_aids = util.take_percentile_parts(all_gt_aids, top_gt, mid_gt,
                                                 bot_gt)
            gf_aids = util.take_percentile_parts(all_gf_aids, top_gf, mid_gf,
                                                 bot_gf)
            # get unscored examples
            unscored_gt_aids = [aid for aid in qreq_.daids[cm.qnid == dnids]
                                if aid not in cm.daid2_idx]
            rand_gt_aids = util.random_sample(unscored_gt_aids, rand_gt, rng=rng)
            # gf_aids = cm.get_groundfalse_daids()
            _gf_aids = qreq_.daids[cm.qnid != dnids]
            _gf_aids = qreq_.daids.compress(cm.qnid != dnids)
            # gf_aids = ibs.get_annot_groundfalse(cm.qaid, daid_list=qreq_.daids)
            rand_gf_aids = util.random_sample(_gf_aids, rand_gf, rng=rng).tolist()
            chosen_daids = list(ub.unique(gt_aids + gf_aids + rand_gf_aids +
                                          rand_gt_aids))
            aid_pairs.extend([(cm.qaid, aid) for aid in chosen_daids if cm.qaid != aid])

        return aid_pairs

    def _get_cm_agg_aid_ranking(infr, cc):
        aid_to_cm = {cm.qaid: cm for cm in infr.cm_list}
        all_scores = ub.ddict(list)
        for qaid in cc:
            cm = aid_to_cm[qaid]
            # should we be doing nids?
            for daid, score in zip(cm.get_top_aids(), cm.get_top_scores()):
                all_scores[daid].append(score)

        max_scores = sorted((max(scores), aid)
                            for aid, scores in all_scores.items())[::-1]
        ranked_aids = util.take_column(max_scores, 1)
        return ranked_aids

    def _get_cm_edge_data(infr, edges, cm_list=None):
        symmetric = True

        if cm_list is None:
            cm_list = infr.cm_list
        # Find scores for the edges that exist in the graph
        edge_to_data = ub.ddict(dict)
        aid_to_cm = {cm.qaid: cm for cm in cm_list}
        for u, v in edges:
            if symmetric:
                u, v = e_(u, v)
            cm1 = aid_to_cm.get(u, None)
            cm2 = aid_to_cm.get(v, None)
            scores = []
            ranks = []
            for cm in util.filter_Nones([cm1, cm2]):
                for aid in [u, v]:
                    idx = cm.daid2_idx.get(aid, None)
                    if idx is None:
                        continue
                    score = cm.annot_score_list[idx]
                    rank = cm.get_annot_ranks([aid])[0]
                    scores.append(score)
                    ranks.append(rank)
            if len(scores) == 0:
                score = None
                rank = None
            else:
                # Choose whichever one gave the best score
                idx = util.safe_argmax(scores, nans=False)
                score = scores[idx]
                rank = ranks[idx]
            edge_to_data[(u, v)]['score'] = score
            edge_to_data[(u, v)]['rank'] = rank
        return edge_to_data

    def apply_match_scores(infr):
        """

        Applies precomputed matching scores to edges that already exist in the
        graph. Typically you should run infr.apply_match_edges() before running
        this.

        Example:
            >>> # ENABLE_DOCTEST
            >>> infr = testdata_infr('PZ_MTEST')
            >>> infr.exec_matching()
            >>> infr.apply_match_edges()
            >>> infr.apply_match_scores()
            >>> infr.get_edge_attrs('score')
        """
        if infr.cm_list is None:
            infr.print('apply_match_scores - no scores to apply!')
            return
        infr.print('apply_match_scores', 1)
        edges = list(infr.graph.edges())
        edge_to_data = infr._get_cm_edge_data(edges)

        # Remove existing attrs
        util.nx_delete_edge_attr(infr.graph, 'score')
        util.nx_delete_edge_attr(infr.graph, 'rank')
        util.nx_delete_edge_attr(infr.graph, 'normscore')

        edges = list(edge_to_data.keys())
        edge_scores = list(util.take_column(edge_to_data.values(), 'score'))
        edge_scores = util.replace_nones(edge_scores, np.nan)
        edge_scores = np.array(edge_scores)
        edge_ranks = np.array(util.take_column(edge_to_data.values(), 'rank'))
        # take the inf-norm
        normscores = edge_scores / util.safe_max(edge_scores, nans=False)

        # Add new attrs
        infr.set_edge_attrs('score', ub.dzip(edges, edge_scores))
        infr.set_edge_attrs('rank', ub.dzip(edges, edge_ranks))

        # Hack away zero probabilites
        # probs = np.vstack([p_nomatch, p_match, p_notcomp]).T + 1e-9
        # probs = util.normalize(probs, axis=1, ord=1, out=probs)
        # entropy = -(np.log2(probs) * probs).sum(axis=1)
        infr.set_edge_attrs('normscore', dict(zip(edges, normscores)))


class InfrLearning(object):

    def learn_deploy_verifiers(infr, publish=False):
        """
        Uses current knowledge to train verifiers for new unseen pairs.

        Example:
            >>> import ibeis
            >>> ibs = ibeis.opendb('PZ_MTEST')
            >>> infr = ibeis.AnnotInference(ibs, aids='all')
            >>> infr.ensure_mst()
            >>> publish = False
            >>> infr.learn_deploy_verifiers()

        Ignore:
            publish = True
        """
        infr.print('learn_deploy_verifiers')
        from ibeis.algo.verif import vsone
        pblm = vsone.OneVsOneProblem(infr, verbose=True)
        pblm.primary_task_key = 'match_state'
        pblm.default_clf_key = 'RF'
        pblm.default_data_key = 'learn(sum,glob)'
        pblm.setup()
        dpath = '.'

        task_key = 'match_state'
        pblm.deploy(dpath, task_key=task_key, publish=publish)

        task_key = 'photobomb_state'
        if task_key in pblm.eval_task_keys:
            pblm.deploy(dpath, task_key=task_key)

    def learn_evaluation_verifiers(infr):
        """
        Creates a cross-validated ensemble of classifiers to evaluate
        verifier error cases and groundtruth errors.

        Doctest:
            >>> import ibeis
            >>> infr = ibeis.AnnotInference(
            >>>     'PZ_MTEST', aids='all', autoinit='annotmatch',
            >>>     verbose=4)
            >>> verifiers = infr.learn_evaluation_verifiers()
            >>> edges = list(infr.edges())
            >>> verif = verifiers['match_state']
            >>> probs = verif.predict_proba_df(edges)
            >>> print(probs)
        """
        infr.print('learn_evaluataion_verifiers')
        from ibeis.algo.verif import vsone
        pblm = vsone.OneVsOneProblem(infr, verbose=5)
        pblm.primary_task_key = 'match_state'
        pblm.eval_clf_keys = ['RF']
        pblm.eval_data_keys = ['learn(sum,glob)']
        pblm.setup_evaluation()
        if True:
            pblm.report_evaluation()
        verifiers = pblm._make_evaluation_verifiers(pblm.eval_task_keys)
        return verifiers

    def load_published(infr):
        """
        Downloads, caches, and loads pre-trained verifiers.
        This is the default action.
        """
        from ibeis.algo.verif import deploy
        ibs = infr.ibs
        species = ibs.get_primary_database_species(infr.aids)
        infr.verifiers = deploy.Deployer().load_published(ibs, species)

    def load_latest_classifiers(infr, dpath):
        from ibeis.algo.verif import deploy
        import pickle
        task_clf_fpaths = deploy.Deployer(dpath).find_latest_local()
        classifiers = {}
        for task_key, fpath in task_clf_fpaths.items():
            clf_info = pickle.load(open(fpath, 'rb'))
            assert clf_info['metadata']['task_key'] == task_key, (
                'bad saved clf at fpath={}'.format(fpath))
            classifiers[task_key] = clf_info
        infr.verifiers = classifiers
        # return classifiers

    def photobomb_samples(infr):
        edges = list(infr.edges())
        tags_list = list(infr.gen_edge_values('tags', edges=edges, default=[]))
        flags = util.filterflags_general_tags(tags_list, has_any=['photobomb'])
        pb_edges = list(ub.compress(edges, flags))
        return pb_edges


class CandidateSearch(object):
    """ Search for candidate edges """
    def find_lnbnn_candidate_edges(infr):
        """

        Example:
            >>> # ENABLE_DOCTEST
            >>> from graphid import demo
            >>> infr = demo.demodata_mtest_infr()
            >>> cand_edges = infr.find_lnbnn_candidate_edges()
            >>> assert len(cand_edges) > 200
        """
        # Refresh the name labels

        # TODO: abstract into a Ranker class

        # do LNBNN query for new edges
        # Use one-vs-many to establish candidate edges to classify
        infr.exec_matching(name_method='edge', cfgdict={
            'resize_dim': 'width',
            'dim_size': 700,
            'requery': True,
            'can_match_samename': False,
            'can_match_sameimg': False,
            # 'sv_on': False,
        })
        # infr.apply_match_edges(review_cfg={'ranks_top': 5})
        ranks_top = infr.params['ranking.ntop']
        lnbnn_results = set(infr._cm_breaking(review_cfg={'ranks_top': ranks_top}))

        candidate_edges = {
            edge for edge, state in
            zip(lnbnn_results, infr.edge_decision_from(lnbnn_results))
            if state == UNREV
        }

        infr.print('ranking alg found {}/{} unreviewed edges'.format(
            len(candidate_edges), len(lnbnn_results)), 1)

        return candidate_edges

    def ensure_task_probs(infr, edges):
        """
        Ensures that probabilities are assigned to the edges.
        This gaurentees that infr.task_probs contains data for edges.
        (Currently only the primary task is actually ensured)

        Doctest:
            >>> import ibeis
            >>> infr = ibeis.AnnotInference('PZ_MTEST', aids='all',
            >>>                             autoinit='staging')
            >>> edges = list(infr.edges())[0:3]
            >>> infr.load_published()
            >>> assert len(infr.task_probs['match_state']) == 0
            >>> infr.ensure_task_probs(edges)
            >>> assert len(infr.task_probs['match_state']) == 3
            >>> infr.ensure_task_probs(edges)
            >>> assert len(infr.task_probs['match_state']) == 3

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
            task_probs = infr._make_task_probs(need_edges)
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
            >>> import ibeis
            >>> ibs = ibeis.opendb('PZ_MTEST')
            >>> infr = ibeis.AnnotInference(ibs, aids='all')
            >>> infr.ensure_mst()
            >>> priority_edges = list(infr.edges())[0:1]
            >>> infr.ensure_priority_scores(priority_edges)

        Doctest:
            >>> import ibeis
            >>> ibs = ibeis.opendb('PZ_MTEST')
            >>> infr = ibeis.AnnotInference(ibs, aids='all')
            >>> infr.ensure_mst()
            >>> infr.load_published()
            >>> priority_edges = list(infr.edges())
            >>> infr.ensure_priority_scores(priority_edges)

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

    def ensure_prioritized(infr, priority_edges):
        priority_edges = list(priority_edges)
        metric, priority = infr.ensure_priority_scores(priority_edges)
        infr.prioritize(metric=metric, edges=priority_edges, scores=priority)

    def add_candidate_edges(infr, candidate_edges):
        candidate_edges = list(candidate_edges)
        new_edges = infr.ensure_edges_from(candidate_edges)

        if infr.test_mode:
            assert False, 'cannot apply edge truth (forgot what that did)'
            infr.apply_edge_truth(new_edges)

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

    def refresh_candidate_edges(infr):
        """
        Search for candidate edges.
        Assign each edge a priority and add to queue.
        """
        infr.print('refresh_candidate_edges', 1)
        infr.assert_consistency_invariant()

        if infr.ibs is not None:
            candidate_edges = infr.find_lnbnn_candidate_edges()
        elif hasattr(infr, 'dummy_verif'):
            infr.print('Searching for dummy candidates')
            infr.print('dummy vsone params =' + ub.repr2(
                infr.dummy_verif.dummy_params, nl=1, si=True))
            ranks_top = infr.params['ranking.ntop']
            candidate_edges = infr.dummy_verif.find_candidate_edges(K=ranks_top)
        else:
            raise Exception(
                'No method available to search for candidate edges')
        infr.add_candidate_edges(candidate_edges)
        infr.assert_consistency_invariant()

    def _make_task_probs(infr, edges):
        """
        Predict edge probs for each pairwise classifier task
        """
        if infr.verifiers is None:
            raise ValueError('no classifiers exist')
        if not isinstance(infr.verifiers, dict):
            raise NotImplementedError(
                'need to deploy or implement eval prediction')
        task_keys = list(infr.verifiers.keys())
        task_probs = {}
        # infr.print('[make_taks_probs] predict {} for {} edges'.format(
        for task_key in task_keys:
            infr.print('predict {} for {} edges'.format(
                task_key, len(edges)))
            verif = infr.verifiers[task_key]
            probs_df = verif.predict_proba_df(edges)
            task_probs[task_key] = probs_df
        return task_probs

    def _make_lnbnn_scores(infr, edges):
        edge_to_data = infr._get_cm_edge_data(edges)
        edges = list(edge_to_data.keys())
        edge_scores = list(util.take_column(edge_to_data.values(), 'score'))
        edge_scores = util.replace_nones(edge_scores, np.nan)
        edge_scores = np.array(edge_scores)
        # take the inf-norm
        normscores = edge_scores / util.safe_max(edge_scores, nans=False)
        return normscores


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/graphid/graphid/ibeis/mixin_matching.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)