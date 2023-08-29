import itertools as it
import networkx as nx
import numpy as np
from graphid import util
from graphid.util import nx_utils as nxu
from graphid.core.state import POSTV, NEGTV, INCMP, UNREV
from graphid.core.state import SAME, DIFF, NULL  # NOQA
import ubelt as ub


def demodata_infr(**kwargs):
    """
    Kwargs:
        num_pccs (list): implicit number of individuals
        ccs (list): explicit list of connected components

        p_incon (float): probability a PCC is inconsistent
        p_incomp (float): probability an edge is incomparable
        n_incon (int): target number of inconsistent components (default 3)

        pcc_size_mean (int): average number of annots per PCC
        pcc_size_std (float): std dev of annots per PCC

        pos_redun (int): desired level of positive redundancy

        infer (bool): whether or not to run inference by default default True

        ignore_pair (bool): if True ignores all pairwise dummy edge generation
        p_pair_neg (float): default = .4
        p_pair_incmp (float): default = .2
        p_pair_unrev (float): default = 0.0

    CommandLine:
        python -m graphid.demo.dummy_infr demodata_infr:0 --show
        python -m graphid.demo.dummy_infr demodata_infr:1 --show
        python -m utool.util_inspect recursive_parse_kwargs:2 --mod graphid.demo.dummy_infr --func demodata_infr


    Example:
        >>> from graphid import demo
        >>> import networkx as nx
        >>> kwargs = dict(num_pccs=6, p_incon=.5, size_std=2)
        >>> infr = demo.demodata_infr(**kwargs)
        >>> pccs = list(infr.positive_components())
        >>> assert len(pccs) == kwargs['num_pccs']
        >>> nonfull_pccs = [cc for cc in pccs if len(cc) > 1 and nx.is_empty(nx.complement(infr.pos_graph.subgraph(cc)))]
        >>> expected_n_incon = len(nonfull_pccs) * kwargs['p_incon']
        >>> n_incon = len(list(infr.inconsistent_components()))
        >>> print('status = ' + ub.urepr(infr.status(extended=True)))
        >>> # xdoctest: +REQUIRES(--show)
        >>> infr.show(pickable=True, groupby='name_label')
        >>> util.show_if_requested()

    Doctest:
        >>> from graphid import demo
        >>> import networkx as nx
        >>> kwargs = dict(num_pccs=0)
        >>> infr = demo.demodata_infr(**kwargs)
    """
    from graphid.core.annot_inference import AnnotInference
    from graphid.demo import dummy_algos

    def kwalias(*args):
        params = args[0:-1]
        default = args[-1]
        for key in params:
            if key in kwargs:
                return kwargs[key]
        return default

    num_pccs = kwalias('num_pccs', 16)
    size_mean = kwalias('pcc_size_mean', 'pcc_size', 'size', 5)
    size_std = kwalias('pcc_size_std', 'size_std', 0)
    # p_pcc_incon = kwargs.get('p_incon', .1)
    p_pcc_incon = kwargs.get('p_incon', 0)
    p_pcc_incomp = kwargs.get('p_incomp', 0)
    pcc_sizes = kwalias('pcc_sizes', None)

    pos_redun = kwalias('pos_redun', [1, 2, 3])
    pos_redun = util.ensure_iterable(pos_redun)

    # number of maximum inconsistent edges per pcc
    max_n_incon = kwargs.get('n_incon', 3)

    rng = np.random.RandomState(0)
    counter = 1

    if pcc_sizes is None:
        pcc_sizes = [int(util.randn(size_mean, size_std, rng=rng, a_min=1))
                     for _ in range(num_pccs)]
    else:
        num_pccs = len(pcc_sizes)

    if 'ccs' in kwargs:
        # Overwrites other options
        pcc_sizes = list(map(len, kwargs['ccs']))
        num_pccs = len(pcc_sizes)
        size_mean = None
        size_std = 0

    new_ccs = []
    pcc_iter = list(enumerate(pcc_sizes))
    pcc_iter = ub.ProgIter(pcc_iter, enabled=num_pccs > 20,
                           desc='make pos-demo')
    for i, size in pcc_iter:
        p = .1
        want_connectivity = rng.choice(pos_redun)
        want_connectivity = min(size - 1, want_connectivity)

        # Create basic graph of positive edges with desired connectivity
        g = nxu.random_k_edge_connected_graph(
            size, k=want_connectivity, p=p, rng=rng)
        nx.set_edge_attributes(g, name='evidence_decision', values=POSTV)
        nx.set_edge_attributes(g, name='truth', values=POSTV)
        # nx.set_node_attributes(g, name='orig_name_label', values=i)
        assert nx.is_connected(g)

        # Relabel graph with non-conflicting names
        if 'ccs' in kwargs:
            g = nx.relabel_nodes(g, dict(enumerate(kwargs['ccs'][i])))
        else:
            # Make sure nodes do not conflict with others
            g = nx.relabel_nodes(g, dict(
                enumerate(range(counter, len(g) + counter + 1))))
            counter += len(g)

        # The probability any edge is inconsistent is `p_incon`
        # This is 1 - P(all edges consistent)
        # which means p(edge is consistent) = (1 - p_incon) / N
        complement_edges = util.estarmap(nxu.e_,
                                         nxu.complement_edges(g))
        if len(complement_edges) > 0:
            # compute probability that any particular edge is inconsistent
            # to achieve probability the PCC is inconsistent
            p_edge_inconn = 1 - (1 - p_pcc_incon) ** (1 / len(complement_edges))
            p_edge_unrev = .1
            p_edge_notcomp = 1 - (1 - p_pcc_incomp) ** (1 / len(complement_edges))
            probs = np.array([p_edge_inconn, p_edge_unrev, p_edge_notcomp])
            # if the total probability is greater than 1 the parameters
            # are invalid, so we renormalize to "fix" it.
            # if probs.sum() > 1:
            #     warnings.warn('probabilities sum to more than 1')
            #     probs = probs / probs.sum()
            pcumsum = probs.cumsum()
            # Determine which mutually exclusive state each complement edge is in
            # print('pcumsum = %r' % (pcumsum,))
            states = np.searchsorted(pcumsum, rng.rand(len(complement_edges)))

            incon_idxs = np.where(states == 0)[0]
            if len(incon_idxs) > max_n_incon:
                print('max_n_incon = %r' % (max_n_incon,))
                chosen = rng.choice(incon_idxs, max_n_incon, replace=False)
                states[np.setdiff1d(incon_idxs, chosen)] = len(probs)

            grouped_edges = ub.group_items(complement_edges, states)
            for state, edges in grouped_edges.items():
                truth = POSTV
                if state == 0:
                    # Add in inconsistent edges
                    evidence_decision = NEGTV
                    # TODO: truth could be INCMP or POSTV
                    # new_edges.append((u, v, {'evidence_decision': NEGTV}))
                elif state == 1:
                    evidence_decision = UNREV
                    # TODO: truth could be INCMP or POSTV
                    # new_edges.append((u, v, {'evidence_decision': UNREV}))
                elif state == 2:
                    evidence_decision = INCMP
                    truth = INCMP
                else:
                    continue
                # Add in candidate edges
                attrs = {'evidence_decision': evidence_decision, 'truth': truth}
                for (u, v) in edges:
                    g.add_edge(u, v, **attrs)
        new_ccs.append(g)
        # (list(g.nodes()), new_edges))

    if len(new_ccs) == 0:
        pos_g = nx.Graph()
    else:
        pos_g = nx.union_all(new_ccs)

    assert len(new_ccs) == len(list(nx.connected_components(pos_g)))
    assert num_pccs == len(new_ccs)

    # Add edges between the PCCS
    neg_edges = []

    if not kwalias('ignore_pair', False):
        print('making pairs')

        pair_attrs_lookup = {
            0: {'evidence_decision': NEGTV, 'truth': NEGTV},
            1: {'evidence_decision': INCMP, 'truth': INCMP},
            2: {'evidence_decision': UNREV, 'truth': NEGTV},  # could be incomp or neg
        }

        # These are the probabilities that one edge has this state
        p_pair_neg = kwalias('p_pair_neg', .4)
        p_pair_incmp = kwalias('p_pair_incmp', .2)
        p_pair_unrev = kwalias('p_pair_unrev', 0)

        # p_pair_neg = 1
        cc_combos = ((list(g1.nodes()), list(g2.nodes()))
                     for (g1, g2) in it.combinations(new_ccs, 2))
        valid_cc_combos = [
            (cc1, cc2)
            for cc1, cc2 in cc_combos if len(cc1) and len(cc2)
        ]
        for cc1, cc2 in ub.ProgIter(valid_cc_combos, desc='make neg-demo'):
            possible_edges = util.estarmap(nxu.e_, it.product(cc1, cc2))
            # probability that any edge between these PCCs is negative
            n_edges = len(possible_edges)
            p_edge_neg   = 1 - (1 - p_pair_neg)   ** (1 / n_edges)
            p_edge_incmp = 1 - (1 - p_pair_incmp) ** (1 / n_edges)
            p_edge_unrev = 1 - (1 - p_pair_unrev) ** (1 / n_edges)

            # Create event space with sizes proportional to probabilities
            pcumsum = np.cumsum([p_edge_neg, p_edge_incmp, p_edge_unrev])
            # Roll dice for each of the edge to see which state it lands on
            possible_pstate = rng.rand(len(possible_edges))
            states = np.searchsorted(pcumsum, possible_pstate)

            flags = states < len(pcumsum)
            stateful_states = states.compress(flags)
            stateful_edges = list(ub.compress(possible_edges, flags))

            unique_states, groupxs_list = util.group_indices(stateful_states)
            for state, groupxs in zip(unique_states, groupxs_list):
                # print('state = %r' % (state,))
                # Add in candidate edges
                edges = list(ub.take(stateful_edges, groupxs))
                attrs = pair_attrs_lookup[state]
                for (u, v) in edges:
                    neg_edges.append((u, v, attrs))
        print('Made {} neg_edges between PCCS'.format(len(neg_edges)))
    else:
        print('ignoring pairs')

    G = AnnotInference._graph_cls()
    G.add_nodes_from(pos_g.nodes(data=True))
    G.add_edges_from(pos_g.edges(data=True))
    G.add_edges_from(neg_edges)
    infr = AnnotInference.from_netx(G, infer=kwargs.get('infer', True))
    infr.verbose = 3

    infr.relabel_using_reviews(rectify=False)

    # fontname = 'Ubuntu'
    fontsize = 12
    fontname = 'sans'
    splines = 'spline'
    # splines = 'ortho'
    # splines = 'line'
    infr.set_node_attrs('shape', 'circle')
    infr.graph.graph['ignore_labels'] = True
    infr.graph.graph['dark_background'] = False
    infr.graph.graph['fontname'] = fontname
    infr.graph.graph['fontsize'] = fontsize
    infr.graph.graph['splines'] = splines
    infr.set_node_attrs('width', 29)
    infr.set_node_attrs('height', 29)
    infr.set_node_attrs('fontsize', fontsize)
    infr.set_node_attrs('fontname', fontname)
    infr.set_node_attrs('fixed_size', True)

    # Set synthetic ground-truth attributes for testing
    infr.edge_truth = infr.get_edge_attrs('truth')
    # Make synthetic verif
    dummy_verif = dummy_algos.DummyVerif(infr)
    dummy_ranker = dummy_algos.DummyRanker(dummy_verif)
    infr.set_verifier(dummy_verif)
    infr.set_ranker(dummy_ranker)

    infr.dummy_verif = dummy_verif
    infr.demokw = kwargs
    return infr


if __name__ == '__main__':
    """
    CommandLine:
        python -m graphid.demo.dummy_infr all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
