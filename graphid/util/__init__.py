"""
mkinit graphid.util
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals

__DYNAMIC__ = False
if __DYNAMIC__:
    import mkinit
    exec(mkinit.dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    from graphid.util import mplutil
    from graphid.util import name_rectifier
    from graphid.util import nx_dynamic_graph
    from graphid.util import nx_utils
    from graphid.util import priority_queue
    from graphid.util import util_boxes
    from graphid.util import util_grabdata
    from graphid.util import util_graphviz
    from graphid.util import util_group
    from graphid.util import util_image
    from graphid.util import util_kw
    from graphid.util import util_misc
    from graphid.util import util_numpy
    from graphid.util import util_random
    from graphid.util import util_tags
    from graphid.util.mplutil import (Color, PanEvents, PlotNums, adjust_subplots,
                                      axes_extent, colorbar, deterministic_shuffle,
                                      dict_intersection, distinct_colors,
                                      distinct_markers, draw_border, draw_boxes,
                                      draw_line_segments, ensure_fnum,
                                      extract_axes_extents, figure,
                                      get_axis_xy_width_height, imshow, legend,
                                      make_heatmask, multi_plot, next_fnum,
                                      pan_factory, pandas_plot_matrix, qtensure,
                                      relative_text, reverse_colormap, save_parts,
                                      scores_to_cmap, scores_to_color,
                                      set_figtitle, show_if_requested,
                                      zoom_factory,)
    from graphid.util.name_rectifier import (demodata_oldnames,
                                             find_consistent_labeling,
                                             simple_munkres,)
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
                                       nx_delete_None_edge_attr,
                                       nx_delete_None_node_attr,
                                       nx_delete_edge_attr, nx_delete_node_attr,
                                       nx_edges, nx_gen_edge_attrs,
                                       nx_gen_edge_values, nx_gen_node_attrs,
                                       nx_gen_node_values, nx_node_dict,
                                       random_k_edge_connected_graph, take_column,)
    from graphid.util.priority_queue import (PriorityQueue,)
    from graphid.util.util_boxes import (Boxes, box_ious_py,)
    from graphid.util.util_grabdata import (TESTIMG_URL_DICT, grab_test_imgpath,)
    from graphid.util.util_graphviz import (GRAPHVIZ_KEYS, LARGE_GRAPH,
                                            apply_graph_layout_attrs,
                                            bbox_from_extent, draw_network2,
                                            dump_nx_ondisk, ensure_nonhex_color,
                                            get_explicit_graph,
                                            get_graph_bounding_box, get_nx_layout,
                                            get_pointset_extents, make_agraph,
                                            netx_draw_images_at_positions,
                                            nx_agraph_layout,
                                            nx_ensure_agraph_color,
                                            parse_aedge_layout_attrs,
                                            parse_anode_layout_attrs,
                                            parse_html_graphviz_attrs, parse_point,
                                            patch_pygraphviz, show_nx,
                                            stack_graphs, translate_graph,
                                            translate_graph_to_origin,)
    from graphid.util.util_group import (group_pairs, grouping_delta,
                                         order_dict_by, sort_dict, sortedby,)
    from graphid.util.util_image import (convert_colorspace, ensure_float01,
                                         get_num_channels, imread, imwrite,)
    from graphid.util.util_kw import (KWSpec,)
    from graphid.util.util_misc import (all_dict_combinations, aslist,
                                        classproperty, cprint, delete_dict_keys,
                                        delete_items_by_index, ensure_iterable,
                                        estarmap, flag_None_items, get_timestamp,
                                        highlight_regex, isect, iteritems_sorted,
                                        make_index_lookup, partial_order, randn,
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
