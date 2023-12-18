"""
mkinit graphid.util --lazy_loader_typed
"""
import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

__all__ = ['BLACK', 'Boxes', 'Color', 'DynConnGraph', 'GRAPHVIZ_KEYS',
           'GraphHelperMixin', 'KWSpec', 'LARGE_GRAPH', 'NEUTRAL_BLUE',
           'NiceGraph', 'PanEvents', 'PlotNums', 'PriorityQueue',
           'TESTIMG_URL_DICT', 'adjust_subplots', 'alias_tags',
           'all_dict_combinations', 'apply_graph_layout_attrs',
           'apply_grouping', 'aslist', 'assert_raises', 'atleast_nd',
           'autompl', 'ax_absolute_text', 'axes_extent', 'bbox_from_extent',
           'bfs_conditional', 'box_ious_py', 'build_alias_map',
           'cartoon_stacked_rects', 'classproperty', 'colorbar',
           'complement_edges', 'convert_colorspace', 'cprint',
           'delete_dict_keys', 'delete_items_by_index', 'demodata_bridge',
           'demodata_oldnames', 'demodata_tarjan_bridge',
           'deterministic_shuffle', 'diag_product', 'dict_intersection',
           'dict_take_column', 'distinct_colors', 'distinct_markers',
           'draw_border', 'draw_boxes', 'draw_line_segments', 'draw_network2',
           'dump_nx_ondisk', 'e_', 'edge_df', 'edges_between', 'edges_cross',
           'edges_inside', 'edges_outgoing', 'ensure_float01', 'ensure_fnum',
           'ensure_iterable', 'ensure_multi_index', 'ensure_nonhex_color',
           'ensure_rng', 'estarmap', 'extract_axes_extents', 'figure',
           'filterflags_general_tags', 'find_consistent_labeling',
           'flag_None_items', 'get_axis_xy_width_height', 'get_explicit_graph',
           'get_graph_bounding_box', 'get_num_channels', 'get_nx_layout',
           'get_plotdat_dict', 'get_pointset_extents', 'get_timestamp',
           'grab_test_image_fpath', 'grab_test_imgpath', 'graph_info',
           'group_indices', 'group_items', 'group_name_edges', 'group_pairs',
           'grouping_delta', 'highlight_regex', 'imread', 'imshow', 'imwrite',
           'is_complete', 'is_k_edge_connected', 'isect', 'isect_flags',
           'itake_column', 'iter_reduce_ufunc', 'iteritems_sorted',
           'k_edge_augmentation', 'legend', 'list_roll', 'make_agraph',
           'make_bbox', 'make_heatmask', 'make_index_lookup', 'mpl_plottool',
           'mplutil', 'multi_plot', 'name_rectifier',
           'netx_draw_images_at_positions', 'next_fnum', 'nx_UnionFind',
           'nx_agraph_layout', 'nx_delete_None_edge_attr',
           'nx_delete_None_node_attr', 'nx_delete_edge_attr',
           'nx_delete_node_attr', 'nx_dynamic_graph', 'nx_edges',
           'nx_ensure_agraph_color', 'nx_gen_edge_attrs', 'nx_gen_edge_values',
           'nx_gen_node_attrs', 'nx_gen_node_values', 'nx_node_dict',
           'nx_sink_nodes', 'nx_source_nodes', 'nx_utils', 'order_dict_by',
           'pan_factory', 'pandas_plot_matrix', 'parse_aedge_layout_attrs',
           'parse_anode_layout_attrs', 'parse_fontkw',
           'parse_html_graphviz_attrs', 'parse_point', 'partial_order',
           'patch_pygraphviz', 'priority_queue', 'qtensure', 'randn',
           'random_combinations', 'random_k_edge_connected_graph',
           'random_product', 'regex_word', 'relative_text', 'replace_nones',
           'reverse_colormap', 'safe_argmax', 'safe_extreme', 'safe_max',
           'safe_min', 'save_parts', 'scores_to_cmap', 'scores_to_color',
           'set_figtitle', 'set_plotdat', 'setdiff', 'show_if_requested',
           'show_nx', 'shuffle', 'simple_munkres', 'snapped_slice',
           'sort_dict', 'sortedby', 'stack_graphs', 'stats_dict', 'tag_hist',
           'take_column', 'take_percentile_parts', 'translate_graph',
           'translate_graph_to_origin', 'util_boxes', 'util_grabdata',
           'util_graphviz', 'util_group', 'util_image', 'util_kw', 'util_misc',
           'util_numpy', 'util_random', 'util_tags', 'where', 'zoom_factory']
