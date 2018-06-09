from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub
from graphid.util import nx_utils as nxu
from graphid.core.state import (POSTV, NEGTV, INCMP, UNREV, UNKWN, UNINFERABLE)
from graphid.core.state import (SAME, DIFF, NULL)  # NOQA


class NonDynamicUpdate(object):
    """
    Updates all bookkeeping data structures.

    This procedures defined here are linear in the size of the graph, which
    makes it prohibitive to execute whenever the graph updates (hence why
    dynamic update is important).  However, dynamic update is much tricker to
    implement, so this serves two purposes:
        (1) as a sanity check that the dynamic update is working and
        (2) as a means to initialize all the bookkeeping data structures into a
        good state.
    """

    def apply_nondynamic_update(infr, graph=None):
        """
        Recomputes all dynamic bookkeeping for a graph in any state.
        This ensures that subsequent dyanmic inference can be applied.

        Example:
            >>> from graphid import demo
            >>> num_pccs = 250
            >>> kwargs = dict(num_pccs=100, p_incon=.3)
            >>> infr = demo.demodata_infr(infer=False, **kwargs)
            >>> graph = None
            >>> infr.apply_nondynamic_update()
            >>> infr.assert_neg_metagraph()
        """
        # Cluster edges by category
        ne_to_edges = infr.collapsed_meta_edges()
        categories = infr.categorize_edges(graph, ne_to_edges)

        infr.set_edge_attrs(
            'inferred_state',
            ub.dzip(ub.flatten(categories[POSTV].values()), ['same'])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ub.dzip(ub.flatten(categories[NEGTV].values()), ['diff'])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ub.dzip(ub.flatten(categories[INCMP].values()), [INCMP])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ub.dzip(ub.flatten(categories[UNKWN].values()), [UNKWN])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ub.dzip(ub.flatten(categories[UNREV].values()), [None])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ub.dzip(ub.flatten(categories['inconsistent_internal'].values()),
                    ['inconsistent_internal'])
        )
        infr.set_edge_attrs(
            'inferred_state',
            ub.dzip(ub.flatten(categories['inconsistent_external'].values()),
                    ['inconsistent_external'])
        )

        # Ensure bookkeeping is taken care of
        # * positive redundancy
        # * negative redundancy
        # * inconsistency
        infr.pos_redun_nids = set(infr.find_pos_redun_nids())
        infr.neg_redun_metagraph = infr._graph_cls(list(infr.find_neg_redun_nids()))

        # make a node for each PCC, and place an edge between any pccs with at
        # least one negative edge, with weight being the number of negative
        # edges. Self loops indicate inconsistency.
        infr.neg_metagraph = infr._graph_cls()
        infr.neg_metagraph.add_nodes_from(infr.pos_graph.component_labels())
        for (nid1, nid2), edges in ne_to_edges[NEGTV].items():
            infr.neg_metagraph.add_edge(nid1, nid2, weight=len(edges))

        infr.recover_graph.clear()
        nid_to_errors = {}
        for nid, intern_edges in categories['inconsistent_internal'].items():
            cc = infr.pos_graph.component_nodes(nid)
            pos_subgraph = infr.pos_graph.subgraph(cc, dynamic=False).copy()
            neg_edges = list(nxu.edges_inside(infr.neg_graph, cc))
            recover_hypothesis = dict(infr.hypothesis_errors(pos_subgraph,
                                                             neg_edges))
            nid_to_errors[nid] = set(recover_hypothesis.keys())
            infr.recover_graph.add_edges_from(pos_subgraph.edges())

        # Delete old hypothesis
        infr.set_edge_attrs(
            'maybe_error',
            ub.dzip(ub.flatten(infr.nid_to_errors.values()), [None])
        )
        # Set new hypothesis
        infr.set_edge_attrs(
            'maybe_error',
            ub.dzip(ub.flatten(nid_to_errors.values()), [True])
        )
        infr.nid_to_errors = nid_to_errors

        # no longer dirty
        if graph is None:
            infr.dirty = False

    def collapsed_meta_edges(infr, graph=None):
        """
        Collapse the grah such that each PCC is a node. Get a list of edges
        within/between each PCC.
        """
        states = (POSTV, NEGTV, INCMP, UNREV, UNKWN)
        rev_graph = {key: infr.review_graphs[key] for key in states}
        if graph is None or graph is infr.graph:
            graph = infr.graph
            nodes = None
        else:
            # Need to extract relevant subgraphs
            nodes = list(graph.nodes())
            for key in states:
                if key == POSTV:
                    rev_graph[key] = rev_graph[key].subgraph(nodes,
                                                             dynamic=False)
                else:
                    rev_graph[key] = rev_graph[key].subgraph(nodes)

        # TODO: Rebalance union find to ensure parents is a single lookup
        # infr.pos_graph._union_find.rebalance(nodes)
        # node_to_label = infr.pos_graph._union_find.parents
        node_to_label = infr.pos_graph._union_find

        # Get reviewed edges using fast lookup structures
        ne_to_edges = {
            key: nxu.group_name_edges(rev_graph[key], node_to_label)
            for key in states
        }
        return ne_to_edges

    def categorize_edges(infr, graph=None, ne_to_edges=None):
        """
        Non-dynamically computes the status of each edge in the graph.
        This is can be used to verify the dynamic computations and update when
        the dynamic state is lost.

        Example:
            >>> from graphid import demo
            >>> num_pccs = 250 if ub.argflag('--profile') else 100
            >>> kwargs = dict(num_pccs=100, p_incon=.3)
            >>> infr = demo.demodata_infr(infer=False, **kwargs)
            >>> graph = None
            >>> cat = infr.categorize_edges()
        """
        states = (POSTV, NEGTV, INCMP, UNREV, UNKWN)

        if ne_to_edges is None:
            ne_to_edges = infr.collapsed_meta_edges(graph)

        # Use reviewed edges to determine status of PCCs (repr by name ids)
        # The next steps will rectify duplicates in these sets
        name_edges = {key: set(ne_to_edges[key].keys()) for key in states}

        # Positive and negative decisions override incomparable and unreviewed
        for key in UNINFERABLE:
            name_edges[key].difference_update(name_edges[POSTV])
            name_edges[key].difference_update(name_edges[NEGTV])

        # Negative edges within a PCC signals that an inconsistency exists
        # Remove inconsistencies from the name edges
        incon_internal_ne = name_edges[NEGTV].intersection(name_edges[POSTV])
        name_edges[POSTV].difference_update(incon_internal_ne)
        name_edges[NEGTV].difference_update(incon_internal_ne)

        if __debug__:
            assert all(n1 == n2 for n1, n2 in name_edges[POSTV]), (
                'All positive edges should be internal to a PCC')
            assert len(name_edges[INCMP].intersection(incon_internal_ne)) == 0
            assert len(name_edges[UNREV].intersection(incon_internal_ne)) == 0
            assert len(name_edges[UNKWN].intersection(incon_internal_ne)) == 0
            assert all(n1 == n2 for n1, n2 in incon_internal_ne), (
                'incon_internal edges should be internal to a PCC')

        # External inconsistentices are edges leaving inconsistent components
        incon_internal_nids = {n1 for n1, n2 in incon_internal_ne}
        incon_external_ne = set([])
        # Find all edges leaving an inconsistent PCC
        for key in (NEGTV,) + UNINFERABLE:
            incon_external_ne.update({
                (nid1, nid2) for nid1, nid2 in name_edges[key]
                if nid1 in incon_internal_nids or nid2 in incon_internal_nids
            })
        for key in (NEGTV,) + UNINFERABLE:
            name_edges[key].difference_update(incon_external_ne)

        # Inference between names is now complete.
        # Now we expand this inference and project the labels onto the
        # annotation edges corresponding to each name edge.

        # Version of union that accepts generators
        union = lambda gen: set.union(*gen)  # NOQA

        # Find edges within consistent PCCs
        positive = {
            nid1: union(
                ne_to_edges[key][(nid1, nid2)]
                for key in (POSTV,) + UNINFERABLE)
            for nid1, nid2 in name_edges[POSTV]
        }
        # Find edges between 1-negative-redundant consistent PCCs
        negative = {
            (nid1, nid2): union(
                ne_to_edges[key][(nid1, nid2)]
                for key in (NEGTV,) + UNINFERABLE)
            for nid1, nid2 in name_edges[NEGTV]
        }
        # Find edges internal to inconsistent PCCs
        incon_internal = {
            nid: union(
                ne_to_edges[key][(nid, nid)]
                for key in (POSTV, NEGTV,) + UNINFERABLE)
            for nid in incon_internal_nids
        }
        # Find edges leaving inconsistent PCCs
        incon_external = {
            (nid1, nid2): union(
                ne_to_edges[key][(nid1, nid2)]
                for key in (NEGTV,) + UNINFERABLE)
            for nid1, nid2 in incon_external_ne
        }
        # Unknown names may have been comparable but the reviewer did not
        # know and could not guess. Likely bad quality.
        unknown = {
            (nid1, nid2): ne_to_edges[UNKWN][(nid1, nid2)]
            for (nid1, nid2) in name_edges[UNKWN]
        }
        # Incomparable names cannot make inference about any other edges
        notcomparable = {
            (nid1, nid2): ne_to_edges[INCMP][(nid1, nid2)]
            for (nid1, nid2) in name_edges[INCMP]
        }
        # Unreviewed edges are between any name not known to be negative
        # (this ignores specific incomparable edges)
        unreviewed = {
            (nid1, nid2): ne_to_edges[UNREV][(nid1, nid2)]
            for (nid1, nid2) in name_edges[UNREV]
        }

        ne_categories = {
            POSTV: positive,
            NEGTV: negative,
            UNREV: unreviewed,
            INCMP: notcomparable,
            UNKWN: unknown,
            'inconsistent_internal': incon_internal,
            'inconsistent_external': incon_external,
        }
        return ne_categories
