"""
Implementations of Ranker and Verifier for IBEIS

TODO: merge these into the IBEIS repo and then enable the doctests.
"""
from graphid.core import abstract
import ubelt as ub


class LNBNN_Ranker(abstract.Ranker):
    """
    Wrapper around the IBEIS LNBNN "HotSpotter" Algorithm

    This is derived from the original `AnnotInfrMatching` mixin, but now
    it cleanly separates the functionality of ranking from the graph algo.

    Example:
        >>> # DISABLE_DOCTEST
        >>> import ibeis
        >>> import graphid
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> # Construct an empty graph with PZ_MTEST
        >>> infr = graphid.core.AnnotInference()
        >>> infr.add_aids(ibs.annots().aids)
        >>> # Construct the ranking wrapper
        >>> ranker = LNBNN_Ranker(infr, ibs)
        >>> ranked_results = ranker.predict_single_ranking(1)
        >>> assert 1 not in ranked_results
        >>> assert 2 in ranked_results
        >>> new_edges = ranker.predict_candidate_edges([1, 2])
        >>> assert (1, 2) in new_edges
        >>> assert (2, 1) not in new_edges
    """
    def __init__(ranker, infr, ibs):
        ranker.infr = infr
        ranker.ibs = ibs
        ranker.cfgdict = {
            'resize_dim': 'width',
            'dim_size': 700,
            'requery': True,
            'can_match_samename': False,
            'can_match_sameimg': False,
            'prescore_method': 'csum',
            'score_method': 'csum',
            'K': 3,
            'Knorm': 3,
        }

    def predict_single_ranking(ranker, node, K=10):
        return list(ranker.predict_rankings([node], K=K))[0]

    def predict_rankings(ranker, nodes, K=10):
        """
        Use one-vs-many to establish candidate edges to classify
        """
        # do LNBNN query for new edges
        # infr.apply_match_edges(review_cfg={'ranks_top': 5})
        # TODO: expose other ranking algos like SMK
        rank_algo = 'LNBNN'
        ranker.infr.print('Exec {} ranking algorithm'.format(rank_algo), 1)
        qaids = nodes
        daids = sorted(ranker.infr.graph.nodes)

        custom_nid_lookup = {
            aid: nid for nid, cc in ranker.infr.pos_graph._ccs.items() for aid in cc
        }
        qreq_ = ranker.ibs.new_query_request(qaids, daids,
                                             cfgdict=ranker.cfgdict,
                                             custom_nid_lookup=custom_nid_lookup,
                                             verbose=ranker.infr.verbose >= 2)
        cm_list = qreq_.execute()

        for cm in cm_list:
            score_list = cm.annot_score_list
            rank_list = ub.argsort(score_list)[::-1]
            sortx = ub.argsort(rank_list)
            top_sortx = sortx[:K]
            daid_list = list(ub.take(cm.daid_list, top_sortx))
            ranked_nodes = daid_list
            yield ranked_nodes


class VAMP_Verifier(abstract.Verifier):
    """
    Wrapper around the IBEIS VAMP Algorithm

    This is derived from the original `AnnotInfrMatching` mixin, but now
    it cleanly separates the functionality of ranking from the graph algo.

    Example:
        >>> # DISABLE_DOCTEST
        >>> import ibeis
        >>> import graphid
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> # Construct an empty graph with PZ_MTEST
        >>> infr = graphid.core.AnnotInference()
        >>> infr.add_aids(ibs.annots().aids)
        >>> # Get a pretrained sklearn VAMP classifier
        >>> # (yes, I know its the wrong species, its the only one)
        >>> from ibeis.algo.verif import deploy
        >>> clfs = deploy.Deployer().load_published(ibs, species='zebra_grevys')
        >>> clf = clfs['match_state']
        >>> # Construct the ranking wrapper
        >>> verif = VAMP_Verifier(infr, ibs, clf)
        >>> edges = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 4)]
        >>> probs = verif.predict_proba_df(edges)
        >>> print(probs)  # xdoctest: +IGNORE_WANT
                   nomatch  match  notcomp
        aid1 aid2
        1    2      0.0305 0.9664   0.0031
        2    3      0.2048 0.7640   0.0312
        3    4      0.4773 0.5207   0.0020
        4    5      0.9661 0.0000   0.0339
        1    4      0.2998 0.6785   0.0217
    """

    def __init__(verif, infr, ibs, clf):
        verif.infr = infr
        verif.ibs = ibs
        verif.clf = clf

    def predict_proba_df(verif, edges):
        # The ibeis.algo.verif.verifier.Verifier has a lot of magic in it which
        # knows how to properly extract the pairwise features requried for
        # classification.
        # NOTE: This call might fail if ibeis_cnn is not setup properly.
        # I had to:
        # conda install theano
        # IBEIS_CNN requires Lasagne 2.0
        # pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
        try:
            import ibeis_cnn  # NOQA
        except ImportError:
            raise ImportError('VAMP_Verifier usually requires ibeis_cnn')
        probs = verif.clf.predict_proba_df(edges)
        return probs

    def learn_deploy_verifiers(verif):
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
        from ibeis.algo.verif import vsone
        verif.infr.print('learn_deploy_verifiers')

        aids = sorted(verif.infr.graph.nodes)
        pblm = vsone.OneVsOneProblem.from_aids(verif.ibs, aids, verbose=5)

        # pblm = vsone.OneVsOneProblem(infr, verbose=True)
        pblm.primary_task_key = 'match_state'
        pblm.default_clf_key = 'RF'
        pblm.default_data_key = 'learn(sum,glob)'
        pblm.setup()
        dpath = '.'
        task_key = 'match_state'
        pblm.deploy(dpath, task_key=task_key)

    def learn_evaluation_verifiers(verif):
        """

        TODO: Use this to update the learned model given the current available
        groundtruth.

        Creates a cross-validated ensemble of classifiers to evaluate verifier
        error cases and groundtruth errors.

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
        raise NotImplementedError(
            'The OneVsOneProblem will probably need an update to accept '
            'the new graphid API')
        from ibeis.algo.verif import vsone
        verif.infr.print('learn_evaluataion_verifiers')
        # pblm = vsone.OneVsOneProblem(verif.infr, verbose=5)
        # pblm = vsone.OneVsOneProblem(verif, verbose=5)

        # This might work but requires all changes to be commited to the ibeis
        # backend database.

        aids = sorted(verif.infr.graph.nodes)
        pblm = vsone.OneVsOneProblem.from_aids(verif.ibs, aids, verbose=5)
        pblm.primary_task_key = 'match_state'
        pblm.eval_clf_keys = ['RF']
        pblm.eval_data_keys = ['learn(sum,glob)']
        pblm.setup_evaluation()
        if True:
            pblm.report_evaluation()
        clfs = pblm._make_evaluation_verifiers(pblm.eval_task_keys)
        clf = clfs['match_state']
        return clf
