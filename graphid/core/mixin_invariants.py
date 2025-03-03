"""
These check for certain invariants that should be maintained by the dynamic
data structure.
"""
import itertools as it
import networkx as nx
import ubelt as ub
from graphid.util.nx_utils import e_


DEBUG_INCON = True


class AssertInvariants:

    def assert_edge(infr, edge):
        assert edge[0] < edge[1], (
            'edge={} does not satisfy ordering constraint'.format(edge))

    def assert_invariants(infr, msg=''):
        infr.assert_disjoint_invariant(msg)
        infr.assert_union_invariant(msg)
        infr.assert_consistency_invariant(msg)
        infr.assert_recovery_invariant(msg)
        infr.assert_neg_metagraph()

    def assert_neg_metagraph(infr):
        """
        Checks that the negative metgraph is correctly book-kept.
        """
        # The total weight of all edges in the negative metagraph should equal
        # the total number of negative edges.
        neg_weight = sum(nx.get_edge_attributes(
            infr.neg_metagraph, 'weight').values())
        n_neg_edges = infr.neg_graph.number_of_edges()
        assert neg_weight == n_neg_edges, '{} should equal {}'.format(
            neg_weight, n_neg_edges)

        # Self loops in the negative metagraph should correspond to the number
        # of inconsistent components
        neg_self_loop_nids = sorted([
            ne[0] for ne in list(nx.selfloop_edges(infr.neg_metagraph))])
        incon_nids = sorted(infr.nid_to_errors.keys())
        assert neg_self_loop_nids == incon_nids, '{} should equal {}'.format(
            neg_self_loop_nids, incon_nids)

    def assert_union_invariant(infr, msg=''):
        edge_sets = {
            key: set(it.starmap(e_, graph.edges()))
            for key, graph in infr.review_graphs.items()
        }
        edge_union = set.union(*edge_sets.values())
        all_edges = set(it.starmap(e_, infr.graph.edges()))
        if edge_union != all_edges:
            print('ERROR STATUS DUMP:')
            print(ub.urepr(infr.status()))
            raise AssertionError(
                'edge sets must have full union. Found union=%d vs all=%d' % (
                    len(edge_union), len(all_edges)
                ))

    def assert_disjoint_invariant(infr, msg=''):
        # infr.print('assert_disjoint_invariant', 200)
        edge_sets = {
            key: set(it.starmap(e_, graph.edges()))
            for key, graph in infr.review_graphs.items()
        }
        for es1, es2 in it.combinations(edge_sets.values(), 2):
            assert es1.isdisjoint(es2), 'edge sets must be disjoint'

    def assert_consistency_invariant(infr, msg=''):
        if not DEBUG_INCON:
            return
        # infr.print('assert_consistency_invariant', 200)
        if infr.params['inference.enabled']:
            incon_ccs = list(infr.inconsistent_components())
            if len(incon_ccs) > 0:
                raise AssertionError('The graph is not consistent. ' +
                                     msg)

    def assert_recovery_invariant(infr, msg=''):
        if not DEBUG_INCON:
            return
        # infr.print('assert_recovery_invariant', 200)
        inconsistent_ccs = list(infr.inconsistent_components())
        incon_cc = set(ub.flatten(inconsistent_ccs))  # NOQA
