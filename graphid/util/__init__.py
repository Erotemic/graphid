"""
python -c "import ubelt._internal as a; a.autogen_init('graphid.util')"
python -m netharn
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals

__DYNAMIC__ = True
if __DYNAMIC__:
    from ubelt._internal import dynamic_import
    exec(dynamic_import(__name__))
else:
    # <AUTOGEN_INIT>
    from graphid.util import nx_dynamic_graph
    from graphid.util import nx_edge_augmentation
    from graphid.util import nx_edge_kcomponents
    from graphid.util import nx_utils
    from graphid.util import util_kw
    from graphid.util.nx_dynamic_graph import (DynConnGraph, GraphHelperMixin,
                                               NiceGraph, nx_UnionFind,)
    from graphid.util.nx_edge_augmentation import (MetaEdge, bridge_augmentation,
                                                   collapse, compat_shuffle,
                                                   complement_edges,
                                                   greedy_k_edge_augmentation,
                                                   is_k_edge_connected,
                                                   is_locally_k_edge_connected,
                                                   k_edge_augmentation,
                                                   one_edge_augmentation,
                                                   partial_k_edge_augmentation,
                                                   unconstrained_bridge_augmentation,
                                                   unconstrained_one_edge_augmentation,
                                                   weighted_bridge_augmentation,
                                                   weighted_one_edge_augmentation,)
    from graphid.util.nx_edge_kcomponents import (EdgeComponentAuxGraph,
                                                  bridge_components,
                                                  general_k_edge_subgraphs,
                                                  k_edge_components,
                                                  k_edge_subgraphs,)
    from graphid.util.nx_utils import (complement_edges, demodata_bridge,
                                       demodata_tarjan_bridge, diag_product, e_,
                                       edge_df, edges_between, edges_cross,
                                       edges_inside, edges_outgoing,
                                       ensure_multi_index, group_name_edges,
                                       is_complete, is_k_edge_connected,
                                       k_edge_augmentation,
                                       random_k_edge_connected_graph,)
    from graphid.util.util_kw import (KWSpec,)
