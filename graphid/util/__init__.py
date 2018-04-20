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
    from graphid.util import mplutil
    from graphid.util import nx_dynamic_graph
    from graphid.util import nx_utils
    from graphid.util import priority_queue
    from graphid.util import util_group
    from graphid.util import util_kw
    from graphid.util import util_misc
    from graphid.util import util_numpy
    from graphid.util import util_random
    from graphid.util import util_tags
    from graphid.util.mplutil import (Color, PlotNums, adjust_subplots,
                                      axes_extent, colorbar, colorbar_image,
                                      copy_figure_to_clipboard,
                                      deterministic_shuffle, dict_intersection,
                                      distinct_colors, distinct_markers,
                                      draw_border, draw_boxes, draw_line_segments,
                                      ensure_fnum, extract_axes_extents, figure,
                                      imshow, legend, make_heatmask, multi_plot,
                                      next_fnum, pandas_plot_matrix, qtensure,
                                      render_figure_to_image, reverse_colormap,
                                      save_parts, savefig2, scores_to_cmap,
                                      scores_to_color, set_figtitle,
                                      show_if_requested,)
    from graphid.util.nx_dynamic_graph import (DynConnGraph, GraphHelperMixin,
                                               NiceGraph, nx_UnionFind,)
    from graphid.util.nx_utils import (assert_raises, bfs_conditional,
                                       complement_edges, demodata_bridge,
                                       demodata_tarjan_bridge, diag_product,
                                       dict_take_column, e_, edge_df,
                                       edges_between, edges_cross, edges_inside,
                                       edges_outgoing, ensure_multi_index,
                                       graph_info, group_name_edges, is_complete,
                                       is_k_edge_connected, itake_column,
                                       k_edge_augmentation, list_roll,
                                       nx_delete_edge_attr, nx_delete_node_attr,
                                       nx_gen_node_attrs, nx_gen_node_values,
                                       random_k_edge_connected_graph, take_column,)
    from graphid.util.priority_queue import (PriorityQueue,)
    from graphid.util.util_group import (group_pairs, grouping_delta,
                                         order_dict_by, sort_dict, sortedby,)
    from graphid.util.util_kw import (KWSpec,)
    from graphid.util.util_misc import (all_dict_combinations, aslist,
                                        classproperty, cprint, delete_dict_keys,
                                        delete_items_by_index, ensure_iterable,
                                        estarmap, flag_None_items, get_timestamp,
                                        highlight_regex, isect, iteritems_sorted,
                                        make_index_lookup, partial_order,
                                        regex_word, replace_nones, safe_argmax,
                                        safe_extreme, safe_max, safe_min, setdiff,
                                        snapped_slice, stats_dict,
                                        take_percentile_parts, where,)
    from graphid.util.util_numpy import (apply_grouping, atleast_nd, group_indices,
                                         group_items, isect_flags,
                                         iter_reduce_ufunc,)
    from graphid.util.util_random import (ensure_rng, random_combinations,
                                          random_product, shuffle,)
    from graphid.util.util_tags import (alias_tags, build_alias_map,
                                        filterflags_general_tags, tag_hist,)
