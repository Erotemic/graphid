import numpy as np
import networkx as nx
import itertools as it
import ubelt as ub
import pandas as pd
import functools
import collections


def _dz(a, b):
    a = a.tolist() if isinstance(a, np.ndarray) else list(a)
    b = b.tolist() if isinstance(b, np.ndarray) else list(b)
    return ub.dzip(a, b)


def nx_source_nodes(graph):
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            yield node


def nx_sink_nodes(graph):
    for node in graph.nodes():
        if graph.out_degree(node) == 0:
            yield node


def take_column(list_, colx):
    r"""
    accepts a list of (indexables) and returns a list of indexables
    can also return a list of list of indexables if colx is a list

    Args:
        list_ (list):  list of lists
        colx (int or list): index or key in each sublist get item

    Returns:
        list: list of selected items

    Example0:
        >>> list_ = [['a', 'b'], ['c', 'd']]
        >>> colx = 0
        >>> result = take_column(list_, colx)
        >>> result = ub.urepr(result, nl=False)
        >>> print(result)
        ['a', 'c']

    Example1:
        >>> list_ = [['a', 'b'], ['c', 'd']]
        >>> colx = [1, 0]
        >>> result = take_column(list_, colx)
        >>> result = ub.urepr(result, nl=False)
        >>> print(result)
        [['b', 'a'], ['d', 'c']]

    Example2:
        >>> list_ = [{'spam': 'EGGS', 'ham': 'SPAM'}, {'spam': 'JAM', 'ham': 'PRAM'}]
        >>> # colx can be a key or list of keys as well
        >>> colx = ['spam']
        >>> result = take_column(list_, colx)
        >>> result = ub.urepr(result, nl=False)
        >>> print(result)
        [['EGGS'], ['JAM']]
    """
    return list(itake_column(list_, colx))


def dict_take_column(list_of_dicts_, colkey, default=None):
    return [dict_.get(colkey, default) for dict_ in list_of_dicts_]


def itake_column(list_, colx):
    """ iterator version of get_list_column """
    if isinstance(colx, list):
        # multi select
        return ([row[colx_] for colx_ in colx] for row in list_)
    else:
        return (row[colx] for row in list_)


def list_roll(list_, n):
    """
    Like numpy.roll for python lists

    Args:
        list_ (list):
        n (int):

    Returns:
        list:

    References:
        http://stackoverflow.com/questions/9457832/python-list-rotation

    Example:
        >>> list_ = [1, 2, 3, 4, 5]
        >>> n = 2
        >>> result = list_roll(list_, n)
        >>> print(result)
        [4, 5, 1, 2, 3]

    Ignore:
        np.roll(list_, n)
    """
    return list_[-n:] + list_[:-n]


def diag_product(s1, s2):
    """ Does product, but iterates over the diagonal first """
    s1 = list(s1)
    s2 = list(s2)
    if len(s1) > len(s2):
        for _ in range(len(s1)):
            for a, b in zip(s1, s2):
                yield (a, b)
            s1 = list_roll(s1, 1)
    else:
        for _ in range(len(s2)):
            for a, b in zip(s1, s2):
                yield (a, b)
            s2 = list_roll(s2, 1)


def e_(u, v):
    return (u, v) if u < v else (v, u)


def edges_inside(graph, nodes):
    """
    Finds edges within a set of nodes
    Running time is O(len(nodes) ** 2)

    Args:
        graph (nx.Graph): an undirected graph
        nodes1 (set): a set of nodes
    """
    result = set([])
    upper = nodes.copy()
    graph_adj = graph.adj
    for u in nodes:
        for v in upper.intersection(graph_adj[u]):
            result.add(e_(u, v))
        upper.remove(u)
    return result


def edges_outgoing(graph, nodes):
    """
    Finds edges leaving a set of nodes.
    Average running time is O(len(nodes) * ave_degree(nodes))
    Worst case running time is O(G.number_of_edges()).

    Args:
        graph (nx.Graph): a graph
        nodes (set): set of nodes

    Example:
        >>> G = demodata_bridge()
        >>> nodes = {1, 2, 3, 4}
        >>> outgoing = edges_outgoing(G, nodes)
        >>> assert outgoing == {(3, 5), (4, 8)}
    """
    if not isinstance(nodes, set):
        nodes = set(nodes)
    return {e_(u, v) for u in nodes for v in graph.adj[u] if v not in nodes}


def edges_cross(graph, nodes1, nodes2):
    """
    Finds edges between two sets of disjoint nodes.
    Running time is O(len(nodes1) * len(nodes2))

    Args:
        graph (nx.Graph): an undirected graph
        nodes1 (set): set of nodes disjoint from `nodes2`
        nodes2 (set): set of nodes disjoint from `nodes1`.
    """
    return {e_(u, v) for u in nodes1
            for v in nodes2.intersection(graph.adj[u])}


def edges_between(graph, nodes1, nodes2=None, assume_disjoint=False,
                  assume_dense=True):
    r"""
    Get edges between two components or within a single component

    Args:
        graph (nx.Graph): the graph
        nodes1 (set): list of nodes
        nodes2 (set): if None it is equivlanet to nodes2=nodes1 (default=None)
        assume_disjoint (bool): skips expensive check to ensure edges arnt
            returned twice (default=False)

    Example:
        >>> edges = [
        >>>     (1, 2), (2, 3), (3, 4), (4, 1), (4, 3),  # cc 1234
        >>>     (1, 5), (7, 2), (5, 1),  # cc 567 / 5678
        >>>     (7, 5), (5, 6), (8, 7),
        >>> ]
        >>> digraph = nx.DiGraph(edges)
        >>> graph = nx.Graph(edges)
        >>> nodes1 = [1, 2, 3, 4]
        >>> nodes2 = [5, 6, 7]
        >>> n2 = sorted(edges_between(graph, nodes1, nodes2))
        >>> n4 = sorted(edges_between(graph, nodes1))
        >>> n5 = sorted(edges_between(graph, nodes1, nodes1))
        >>> n1 = sorted(edges_between(digraph, nodes1, nodes2))
        >>> n3 = sorted(edges_between(digraph, nodes1))
        >>> print('n2 == %r' % (n2,))
        >>> print('n4 == %r' % (n4,))
        >>> print('n5 == %r' % (n5,))
        >>> print('n1 == %r' % (n1,))
        >>> print('n3 == %r' % (n3,))
        >>> assert n2 == ([(1, 5), (2, 7)]), '2'
        >>> assert n4 == ([(1, 2), (1, 4), (2, 3), (3, 4)]), '4'
        >>> assert n5 == ([(1, 2), (1, 4), (2, 3), (3, 4)]), '5'
        >>> assert n1 == ([(1, 5), (5, 1), (7, 2)]), '1'
        >>> assert n3 == ([(1, 2), (2, 3), (3, 4), (4, 1), (4, 3)]), '3'
        >>> n6 = sorted(edges_between(digraph, nodes1 + [6], nodes2 + [1, 2], assume_dense=False))
        >>> print('n6 = %r' % (n6,))
        >>> n6 = sorted(edges_between(digraph, nodes1 + [6], nodes2 + [1, 2], assume_dense=True))
        >>> print('n6 = %r' % (n6,))
        >>> assert n6 == ([(1, 2), (1, 5), (2, 3), (4, 1), (5, 1), (5, 6), (7, 2)]), '6'
    """
    if assume_dense:
        edges = _edges_between_dense(graph, nodes1, nodes2, assume_disjoint)
    else:
        edges = _edges_between_sparse(graph, nodes1, nodes2, assume_disjoint)
    if graph.is_directed():
        for u, v in edges:
            yield u, v
    else:
        for u, v in edges:
            yield e_(u, v)


def _edges_between_dense(graph, nodes1, nodes2=None, assume_disjoint=False):
    """
    The dense method is where we enumerate all possible edges and just take the
    ones that exist (faster for very dense graphs)
    """
    if nodes2 is None or nodes2 is nodes1:
        # Case where we are looking at internal nodes only
        edge_iter = it.combinations(nodes1, 2)
    elif assume_disjoint:
        # We assume len(isect(nodes1, nodes2)) == 0
        edge_iter = it.product(nodes1, nodes2)
    else:
        # make sure a single edge is not returned twice
        # in the case where len(isect(nodes1, nodes2)) > 0
        if not isinstance(nodes1, set):
            nodes1 = set(nodes1)
        if not isinstance(nodes2, set):
            nodes2 = set(nodes2)
        nodes_isect = nodes1.intersection(nodes2)
        nodes_only1 = nodes1 - nodes_isect
        nodes_only2 = nodes2 - nodes_isect
        edge_sets = [it.product(nodes_only1, nodes_only2),
                     it.product(nodes_only1, nodes_isect),
                     it.product(nodes_only2, nodes_isect),
                     it.combinations(nodes_isect, 2)]
        edge_iter = it.chain.from_iterable(edge_sets)

    if graph.is_directed():
        for n1, n2 in edge_iter:
            if graph.has_edge(n1, n2):
                yield n1, n2
            if graph.has_edge(n2, n1):
                yield n2, n1
    else:
        for n1, n2 in edge_iter:
            if graph.has_edge(n1, n2):
                yield n1, n2


def _edges_inside_lower(graph, both_adj):
    """ finds lower triangular edges inside the nodes """
    both_lower = set([])
    for u, neighbs in both_adj.items():
        neighbsBB_lower = neighbs.intersection(both_lower)
        for v in neighbsBB_lower:
            yield (u, v)
        both_lower.add(u)


def _edges_inside_upper(graph, both_adj):
    """ finds upper triangular edges inside the nodes """
    both_upper = set(both_adj.keys())
    for u, neighbs in both_adj.items():
        neighbsBB_upper = neighbs.intersection(both_upper)
        for v in neighbsBB_upper:
            yield (u, v)
        both_upper.remove(u)


def _edges_between_disjoint(graph, only1_adj, only2):
    """ finds edges between disjoint nodes """
    for u, neighbs in only1_adj.items():
        # Find the neighbors of u in only1 that are also in only2
        neighbs12 = neighbs.intersection(only2)
        for v in neighbs12:
            yield (u, v)


def _edges_between_sparse(graph, nodes1, nodes2=None, assume_disjoint=False):
    """
    In this version we check the intersection of existing edges and the edges
    in the second set (faster for sparse graphs)
    """
    # Notes:
    # 1 = edges only in `nodes1`
    # 2 = edges only in `nodes2`
    # B = edges only in both `nodes1` and `nodes2`

    # Test for special cases
    if nodes2 is None or nodes2 is nodes1:
        # Case where we just are finding internal edges
        both = set(nodes1)
        both_adj  = {u: set(graph.adj[u]) for u in both}
        if graph.is_directed():
            edge_sets = (
                _edges_inside_upper(graph, both_adj),  # B-to-B (u)
                _edges_inside_lower(graph, both_adj),  # B-to-B (l)
            )
        else:
            edge_sets = (
                _edges_inside_upper(graph, both_adj),  # B-to-B (u)
            )
    elif assume_disjoint:
        # Case where we find edges between disjoint sets
        if not isinstance(nodes1, set):
            nodes1 = set(nodes1)
        if not isinstance(nodes2, set):
            nodes2 = set(nodes2)
        only1 = nodes1
        only2 = nodes2
        if graph.is_directed():
            only1_adj = {u: set(graph.adj[u]) for u in only1}
            only2_adj = {u: set(graph.adj[u]) for u in only2}
            edge_sets = (
                _edges_between_disjoint(graph, only1, only2),  # 1-to-2
                _edges_between_disjoint(graph, only2, only1),  # 2-to-1
            )
        else:
            only1_adj = {u: set(graph.adj[u]) for u in only1}
            edge_sets = (
                _edges_between_disjoint(graph, only1, only2),  # 1-to-2
            )
    else:
        # Full general case
        if not isinstance(nodes1, set):
            nodes1 = set(nodes1)
        if nodes2 is None:
            nodes2 = nodes1
        elif not isinstance(nodes2, set):
            nodes2 = set(nodes2)
        both = nodes1.intersection(nodes2)
        only1 = nodes1 - both
        only2 = nodes2 - both

        # Precompute all calls to set(graph.adj[u]) to avoid duplicate calls
        only1_adj = {u: set(graph.adj[u]) for u in only1}
        only2_adj = {u: set(graph.adj[u]) for u in only2}
        both_adj  = {u: set(graph.adj[u]) for u in both}
        if graph.is_directed():
            edge_sets = (
                _edges_between_disjoint(graph, only1_adj, only2),  # 1-to-2
                _edges_between_disjoint(graph, only1_adj, both),   # 1-to-B
                _edges_inside_upper(graph, both_adj),              # B-to-B (u)
                _edges_inside_lower(graph, both_adj),              # B-to-B (l)
                _edges_between_disjoint(graph, both_adj, only1),   # B-to-1
                _edges_between_disjoint(graph, both_adj, only2),   # B-to-2
                _edges_between_disjoint(graph, only2_adj, both),   # 2-to-B
                _edges_between_disjoint(graph, only2_adj, only1),  # 2-to-1
            )
        else:
            edge_sets = (
                _edges_between_disjoint(graph, only1_adj, only2),  # 1-to-2
                _edges_between_disjoint(graph, only1_adj, both),   # 1-to-B
                _edges_inside_upper(graph, both_adj),              # B-to-B (u)
                _edges_between_disjoint(graph, only2_adj, both),   # 2-to-B
            )

    for u, v in it.chain.from_iterable(edge_sets):
        yield u, v


def group_name_edges(g, node_to_label):
    ne_to_edges = ub.ddict(set)
    for u, v in g.edges():
        name_edge = e_(node_to_label[u], node_to_label[v])
        ne_to_edges[name_edge].add(e_(u, v))
    return ne_to_edges


def ensure_multi_index(index, names):
    import pandas as pd
    if not isinstance(index, (pd.MultiIndex, pd.Index)):
        names = ('aid1', 'aid2')
        if len(index) == 0:
            index = pd.MultiIndex([[], []], [[], []], names=names)
        else:
            index = pd.MultiIndex.from_tuples(index, names=names)
    return index


def demodata_bridge():
    # define 2-connected compoments and bridges
    cc2 = [(1, 2, 4, 3, 1, 4), (8, 9, 10, 8), (11, 12, 13, 11)]
    bridges = [(4, 8), (3, 5), (20, 21), (22, 23, 24)]
    G = nx.Graph(ub.flatten(ub.iter_window(path, 2) for path in cc2 + bridges))
    return G


def demodata_tarjan_bridge():
    """
    Example:
        >>> from graphid import util
        >>> G = demodata_tarjan_bridge()
        >>> # xdoc: +REQUIRES(--show)
        >>> util.show_nx(G)
        >>> util.show_if_requested()
    """
    # define 2-connected compoments and bridges
    cc2 = [(1, 2, 4, 3, 1, 4), (5, 6, 7, 5), (8, 9, 10, 8),
             (17, 18, 16, 15, 17), (11, 12, 14, 13, 11, 14)]
    bridges = [(4, 8), (3, 5), (3, 17)]
    G = nx.Graph(ub.flatten(ub.iter_window(path, 2) for path in cc2 + bridges))
    return G


# def is_tri_edge_connected(G):
#     """
#     Yet another Simple Algorithm for Triconnectivity
#     http://www.sciencedirect.com/science/article/pii/S1570866708000415
#     """
#     pass


def is_k_edge_connected(G, k):
    return nx.is_k_edge_connected(G, k)


def complement_edges(G):
    from networkx.algorithms.connectivity.edge_augmentation import complement_edges
    return it.starmap(e_, complement_edges(G))


def k_edge_augmentation(G, k, avail=None, partial=False):
    return it.starmap(e_, nx.k_edge_augmentation(G, k, avail=avail,
                                                 partial=partial))


def is_complete(G, self_loops=False):
    assert not G.is_multigraph()
    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()
    if G.is_directed():
        n_need = (n_nodes * (n_nodes - 1))
    else:
        n_need = (n_nodes * (n_nodes - 1)) // 2
    if self_loops:
        n_need += n_nodes
    return n_edges == n_need


def random_k_edge_connected_graph(size, k, p=.1, rng=None):
    """
    Super hacky way of getting a random k-connected graph

    Example:
        >>> from graphid import util
        >>> size, k, p = 25, 3, .1
        >>> rng = util.ensure_rng(0)
        >>> gs = []
        >>> for x in range(4):
        >>>     G = random_k_edge_connected_graph(size, k, p, rng)
        >>>     gs.append(G)
        >>> # xdoc: +REQUIRES(--show)
        >>> pnum_ = util.PlotNums(nRows=2, nSubplots=len(gs))
        >>> fnum = 1
        >>> for g in gs:
        >>>     util.show_nx(g, fnum=fnum, pnum=pnum_())
    """
    for count in it.count(0):
        seed = None if rng is None else rng.randint((2 ** 31 - 1))
        # Randomly generate a graph
        g = nx.fast_gnp_random_graph(size, p, seed=seed)
        conn = nx.edge_connectivity(g)
        # If it has exactly the desired connectivity we are one
        if conn == k:
            break
        # If it has more, then we regenerate the graph with fewer edges
        elif conn > k:
            p = p / 2
        # If it has less then we add a small set of edges to get there
        elif conn < k:
            # p = 2 * p - p ** 2
            # if count == 2:
            aug_edges = list(k_edge_augmentation(g, k))
            g.add_edges_from(aug_edges)
            break
    return g


def edge_df(graph, edges, ignore=None):
    edge_dict = {e: graph.get_edge_data(*e) for e in edges}
    df = pd.DataFrame.from_dict(edge_dict, orient='index')

    if len(df):
        if ignore:
            ignore = df.columns.intersection(ignore)
            df = df.drop(ignore, axis=1)
        try:
            df.index.names = ('u', 'v')
        except Exception:
            pass
    return df


def nx_delete_node_attr(graph, name, nodes=None):
    """
    Removes node attributes

    Doctest:
        >>> G = nx.karate_club_graph()
        >>> nx.set_node_attributes(G, name='foo', values='bar')
        >>> datas = nx.get_node_attributes(G, 'club')
        >>> assert len(nx.get_node_attributes(G, 'club')) == 34
        >>> assert len(nx.get_node_attributes(G, 'foo')) == 34
        >>> nx_delete_node_attr(G, ['club', 'foo'], nodes=[1, 2])
        >>> assert len(nx.get_node_attributes(G, 'club')) == 32
        >>> assert len(nx.get_node_attributes(G, 'foo')) == 32
        >>> nx_delete_node_attr(G, ['club'])
        >>> assert len(nx.get_node_attributes(G, 'club')) == 0
        >>> assert len(nx.get_node_attributes(G, 'foo')) == 32
    """
    if nodes is None:
        nodes = list(graph.nodes())
    removed = 0
    # names = [name] if not isinstance(name, list) else name
    node_dict = graph.nodes

    if isinstance(name, list):
        for node in nodes:
            for name_ in name:
                try:
                    del node_dict[node][name_]
                    removed += 1
                except KeyError:
                    pass
    else:
        for node in nodes:
            try:
                del node_dict[node][name]
                removed += 1
            except KeyError:
                pass
    return removed


def nx_delete_edge_attr(graph, name, edges=None):
    """
    Removes an attributes from specific edges in the graph

    Doctest:
        >>> G = nx.karate_club_graph()
        >>> nx.set_edge_attributes(G, name='spam', values='eggs')
        >>> nx.set_edge_attributes(G, name='foo', values='bar')
        >>> assert len(nx.get_edge_attributes(G, 'spam')) == 78
        >>> assert len(nx.get_edge_attributes(G, 'foo')) == 78
        >>> nx_delete_edge_attr(G, ['spam', 'foo'], edges=[(1, 2)])
        >>> assert len(nx.get_edge_attributes(G, 'spam')) == 77
        >>> assert len(nx.get_edge_attributes(G, 'foo')) == 77
        >>> nx_delete_edge_attr(G, ['spam'])
        >>> assert len(nx.get_edge_attributes(G, 'spam')) == 0
        >>> assert len(nx.get_edge_attributes(G, 'foo')) == 77

    Doctest:
        >>> G = nx.MultiGraph()
        >>> G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (4, 5), (1, 2)])
        >>> nx.set_edge_attributes(G, name='spam', values='eggs')
        >>> nx.set_edge_attributes(G, name='foo', values='bar')
        >>> assert len(nx.get_edge_attributes(G, 'spam')) == 6
        >>> assert len(nx.get_edge_attributes(G, 'foo')) == 6
        >>> nx_delete_edge_attr(G, ['spam', 'foo'], edges=[(1, 2, 0)])
        >>> assert len(nx.get_edge_attributes(G, 'spam')) == 5
        >>> assert len(nx.get_edge_attributes(G, 'foo')) == 5
        >>> nx_delete_edge_attr(G, ['spam'])
        >>> assert len(nx.get_edge_attributes(G, 'spam')) == 0
        >>> assert len(nx.get_edge_attributes(G, 'foo')) == 5
    """
    removed = 0
    keys = [name] if not isinstance(name, (list, tuple)) else name
    if edges is None:
        if graph.is_multigraph():
            edges = graph.edges(keys=True)
        else:
            edges = graph.edges()
    if graph.is_multigraph():
        for u, v, k in edges:
            for key_ in keys:
                try:
                    del graph[u][v][k][key_]
                    removed += 1
                except KeyError:
                    pass
    else:
        for u, v in edges:
            for key_ in keys:
                try:
                    del graph[u][v][key_]
                    removed += 1
                except KeyError:
                    pass
    return removed


def nx_gen_node_values(G, key, nodes, default=ub.NoParam):
    """
    Generates attributes values of specific nodes
    """
    node_dict = G.nodes
    if default is ub.NoParam:
        return (node_dict[n][key] for n in nodes)
    else:
        return (node_dict[n].get(key, default) for n in nodes)


def nx_gen_node_attrs(G, key, nodes=None, default=ub.NoParam,
                      on_missing='error', on_keyerr='default'):
    """
    Improved generator version of nx.get_node_attributes

    Args:
        on_missing (str): Strategy for handling nodes missing from G.
            Can be {'error', 'default', 'filter'}.  defaults to 'error'.
        on_keyerr (str): Strategy for handling keys missing from node dicts.
            Can be {'error', 'default', 'filter'}.  defaults to 'default'
            if default is specified, otherwise defaults to 'error'.

    Notes:
        strategies are:
            error - raises an error if key or node does not exist
            default - returns node, but uses value specified by default
            filter - skips the node

    Example:
        >>> # ENABLE_DOCTEST
        >>> from graphid import util
        >>> G = nx.Graph([(1, 2), (2, 3)])
        >>> nx.set_node_attributes(G, name='part', values={1: 'bar', 3: 'baz'})
        >>> nodes = [1, 2, 3, 4]
        >>> #
        >>> assert len(list(nx_gen_node_attrs(G, 'part', default=None, on_missing='error', on_keyerr='default'))) == 3
        >>> assert len(list(nx_gen_node_attrs(G, 'part', default=None, on_missing='error', on_keyerr='filter'))) == 2
        >>> assert_raises(KeyError, list, nx_gen_node_attrs(G, 'part', on_missing='error', on_keyerr='error'))
        >>> #
        >>> assert len(list(nx_gen_node_attrs(G, 'part', nodes, default=None, on_missing='filter', on_keyerr='default'))) == 3
        >>> assert len(list(nx_gen_node_attrs(G, 'part', nodes, default=None, on_missing='filter', on_keyerr='filter'))) == 2
        >>> assert_raises(KeyError, list, nx_gen_node_attrs(G, 'part', nodes, on_missing='filter', on_keyerr='error'))
        >>> #
        >>> assert len(list(nx_gen_node_attrs(G, 'part', nodes, default=None, on_missing='default', on_keyerr='default'))) == 4
        >>> assert len(list(nx_gen_node_attrs(G, 'part', nodes, default=None, on_missing='default', on_keyerr='filter'))) == 2
        >>> assert_raises(KeyError, list, nx_gen_node_attrs(G, 'part', nodes, on_missing='default', on_keyerr='error'))

    Example:
        >>> # DISABLE_DOCTEST
        >>> # ALL CASES
        >>> from graphid import util
        >>> G = nx.Graph([(1, 2), (2, 3)])
        >>> nx.set_node_attributes(G, name='full', values={1: 'A', 2: 'B', 3: 'C'})
        >>> nx.set_node_attributes(G, name='part', values={1: 'bar', 3: 'baz'})
        >>> nodes = [1, 2, 3, 4]
        >>> attrs = dict(nx_gen_node_attrs(G, 'full'))
        >>> input_grid = {
        >>>     'nodes': [None, (1, 2, 3, 4)],
        >>>     'key': ['part', 'full'],
        >>>     'default': [ub.NoParam, None],
        >>> }
        >>> inputs = util.all_dict_combinations(input_grid)
        >>> kw_grid = {
        >>>     'on_missing': ['error', 'default', 'filter'],
        >>>     'on_keyerr': ['error', 'default', 'filter'],
        >>> }
        >>> kws = util.all_dict_combinations(kw_grid)
        >>> for in_ in inputs:
        >>>     for kw in kws:
        >>>         kw2 = ub.dict_union(kw, in_)
        >>>         #print(kw2)
        >>>         on_missing = kw['on_missing']
        >>>         on_keyerr = kw['on_keyerr']
        >>>         if on_keyerr == 'default' and in_['default'] is ub.NoParam:
        >>>             on_keyerr = 'error'
        >>>         will_miss = False
        >>>         will_keyerr = False
        >>>         if on_missing == 'error':
        >>>             if in_['key'] == 'part' and in_['nodes'] is not None:
        >>>                 will_miss = True
        >>>             if in_['key'] == 'full' and in_['nodes'] is not None:
        >>>                 will_miss = True
        >>>         if on_keyerr == 'error':
        >>>             if in_['key'] == 'part':
        >>>                 will_keyerr = True
        >>>             if on_missing == 'default':
        >>>                 if in_['key'] == 'full' and in_['nodes'] is not None:
        >>>                     will_keyerr = True
        >>>         want_error = will_miss or will_keyerr
        >>>         gen = nx_gen_node_attrs(G, **kw2)
        >>>         try:
        >>>             attrs = list(gen)
        >>>         except KeyError:
        >>>             if not want_error:
        >>>                 raise AssertionError('should not have errored')
        >>>         else:
        >>>             if want_error:
        >>>                 raise AssertionError('should have errored')

    """
    if on_missing is None:
        on_missing = 'error'
    if default is ub.NoParam and on_keyerr == 'default':
        on_keyerr = 'error'
    if nodes is None:
        nodes = G.nodes()
    # Generate `node_data` nodes and data dictionary
    node_dict = G.nodes
    if on_missing == 'error':
        node_data = ((n, node_dict[n]) for n in nodes)
    elif on_missing == 'filter':
        node_data = ((n, node_dict[n]) for n in nodes if n in G)
    elif on_missing == 'default':
        node_data = ((n, node_dict.get(n, {})) for n in nodes)
    else:
        raise KeyError('on_missing={} must be error, filter or default'.format(
            on_missing))
    # Get `node_attrs` desired value out of dictionary
    if on_keyerr == 'error':
        node_attrs = ((n, d[key]) for n, d in node_data)
    elif on_keyerr == 'filter':
        node_attrs = ((n, d[key]) for n, d in node_data if key in d)
    elif on_keyerr == 'default':
        node_attrs = ((n, d.get(key, default)) for n, d in node_data)
    else:
        raise KeyError('on_keyerr={} must be error filter or default'.format(on_keyerr))
    return node_attrs


def graph_info(graph, ignore=None, stats=False, verbose=False):
    from graphid import util
    import pandas as pd

    node_dict = graph.nodes
    node_attrs = list(node_dict.values())
    edge_attrs = list(take_column(graph.edges(data=True), 2))

    if stats:
        node_df = pd.DataFrame(node_attrs)
        edge_df = pd.DataFrame(edge_attrs)
        if ignore is not None:
            util.delete_dict_keys(node_df, ignore)
            util.delete_dict_keys(edge_df, ignore)
        # Not really histograms anymore
        try:
            node_attr_hist = node_df.describe().to_dict()
        except ValueError:
            node_attr_hist
        try:
            edge_attr_hist = edge_df.describe().to_dict()
        except ValueError:
            edge_attr_hist = {}
        key_order = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        node_attr_hist = ub.map_dict_vals(lambda x: util.order_dict_by(x, key_order), node_attr_hist)
        edge_attr_hist = ub.map_dict_vals(lambda x: util.order_dict_by(x, key_order), edge_attr_hist)
    else:
        node_attr_hist = ub.dict_hist(ub.flatten([attr.keys() for attr in node_attrs]))
        edge_attr_hist = ub.dict_hist(ub.flatten([attr.keys() for attr in edge_attrs]))
        if ignore is not None:
            util.delete_dict_keys(edge_attr_hist, ignore)
            util.delete_dict_keys(node_attr_hist, ignore)
    node_type_hist = ub.dict_hist(list(map(type, graph.nodes())))
    info_dict = ub.odict([
        ('directed', graph.is_directed()),
        ('multi', graph.is_multigraph()),
        ('num_nodes', len(graph)),
        ('num_edges', len(list(graph.edges()))),
        ('edge_attr_hist', util.sort_dict(edge_attr_hist)),
        ('node_attr_hist', util.sort_dict(node_attr_hist)),
        ('node_type_hist', util.sort_dict(node_type_hist)),
        ('graph_attrs', graph.graph),
        ('graph_name', graph.name),
    ])
    if verbose:
        print(ub.urepr(info_dict))
    return info_dict


def assert_raises(ex_type, func, *args, **kwargs):
    """
    Checks that a function raises an error when given specific arguments.

    Args:
        ex_type (Exception): exception type
        func (callable): live python function

    Example:
        >>> ex_type = AssertionError
        >>> func = len
        >>> assert_raises(ex_type, assert_raises, ex_type, func, [])
        >>> assert_raises(ValueError, [].index, 0)
    """
    try:
        func(*args, **kwargs)
    except Exception as ex:
        assert isinstance(ex, ex_type), (
            'Raised %r but type should have been %r' % (ex, ex_type))
        return True
    else:
        raise AssertionError('No error was raised')


def bfs_conditional(G, source, reverse=False, keys=True, data=False,
                    yield_nodes=True, yield_if=None,
                    continue_if=None, visited_nodes=None,
                    yield_source=False):
    """
    Produce edges in a breadth-first-search starting at source, but only return
    nodes that satisfiy a condition, and only iterate past a node if it
    satisfies a different condition.

    conditions are callables that take (G, child, edge) and return true or false

    Example:
        >>> import networkx as nx
        >>> G = nx.Graph()
        >>> G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4)])
        >>> continue_if = lambda G, child, edge: True
        >>> result = list(bfs_conditional(G, 1, yield_nodes=False))
        >>> print(result)
        [(1, 2), (1, 3), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (4, 2)]

    Example:
        >>> import networkx as nx
        >>> G = nx.Graph()
        >>> continue_if = lambda G, child, edge: (child % 2 == 0)
        >>> yield_if = lambda G, child, edge: (child % 2 == 1)
        >>> G.add_edges_from([(0, 1), (1, 3), (3, 5), (5, 10),
        >>>                   (4, 3), (3, 6),
        >>>                   (0, 2), (2, 4), (4, 6), (6, 10)])
        >>> result = list(bfs_conditional(G, 0, continue_if=continue_if,
        >>>                                  yield_if=yield_if))
        >>> print(result)
        [1, 3, 5]
    """
    if reverse and hasattr(G, 'reverse'):
        G = G.reverse()
    if isinstance(G, nx.Graph):
        neighbors = functools.partial(G.edges, data=data)
    else:
        neighbors = functools.partial(G.edges, keys=keys, data=data)

    queue = collections.deque([])

    if visited_nodes is None:
        visited_nodes = set([])
    else:
        visited_nodes = set(visited_nodes)

    if source not in visited_nodes:
        if yield_nodes and yield_source:
            yield source
        visited_nodes.add(source)
        new_edges = neighbors(source)
        if isinstance(new_edges, list):
            new_edges = iter(new_edges)
        queue.append((source, new_edges))

    while queue:
        parent, edges = queue[0]
        for edge in edges:
            child = edge[1]
            if yield_nodes:
                if child not in visited_nodes:
                    if yield_if is None or yield_if(G, child, edge):
                        yield child
            else:
                if yield_if is None or yield_if(G, child, edge):
                    yield edge
            if child not in visited_nodes:
                visited_nodes.add(child)
                # Add new children to queue if the condition is satisfied
                if continue_if is None or continue_if(G, child, edge):
                    new_edges = neighbors(child)
                    if isinstance(new_edges, list):
                        new_edges = iter(new_edges)
                    queue.append((child, new_edges))
        queue.popleft()


def nx_gen_edge_attrs(G, key, edges=None, default=ub.NoParam,
                      on_missing='error', on_keyerr='default'):
    """
    Improved generator version of nx.get_edge_attributes

    Args:
        on_missing (str): Strategy for handling nodes missing from G.
            Can be {'error', 'default', 'filter'}.  defaults to 'error'.
            is on_missing is not error, then we allow any edge even if the
            endpoints are not in the graph.
        on_keyerr (str): Strategy for handling keys missing from node dicts.
            Can be {'error', 'default', 'filter'}.  defaults to 'default'
            if default is specified, otherwise defaults to 'error'.

    CommandLine:
        python -m graphid.util.nx_utils nx_gen_edge_attrs

    Example:
        >>> from graphid import util
        >>> from functools import partial
        >>> G = nx.Graph([(1, 2), (2, 3), (3, 4)])
        >>> nx.set_edge_attributes(G, name='part', values={(1, 2): 'bar', (2, 3): 'baz'})
        >>> edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
        >>> func = partial(nx_gen_edge_attrs, G, 'part', default=None)
        >>> #
        >>> assert len(list(func(on_missing='error', on_keyerr='default'))) == 3
        >>> assert len(list(func(on_missing='error', on_keyerr='filter'))) == 2
        >>> util.assert_raises(KeyError, list, func(on_missing='error', on_keyerr='error'))
        >>> #
        >>> assert len(list(func(edges, on_missing='filter', on_keyerr='default'))) == 3
        >>> assert len(list(func(edges, on_missing='filter', on_keyerr='filter'))) == 2
        >>> util.assert_raises(KeyError, list, func(edges, on_missing='filter', on_keyerr='error'))
        >>> #
        >>> assert len(list(func(edges, on_missing='default', on_keyerr='default'))) == 4
        >>> assert len(list(func(edges, on_missing='default', on_keyerr='filter'))) == 2
        >>> util.assert_raises(KeyError, list, func(edges, on_missing='default', on_keyerr='error'))
    """
    if on_missing is None:
        on_missing = 'error'
    if default is ub.NoParam and on_keyerr == 'default':
        on_keyerr = 'error'

    if edges is None:
        if G.is_multigraph():
            raise NotImplementedError('')
            # uvk_iter = G.edges(keys=True)
        else:
            edges = G.edges()
    # Generate `edge_data` edges and data dictionary
    if on_missing == 'error':
        edge_data = (((u, v), G.adj[u][v]) for u, v in edges)
    elif on_missing == 'filter':
        edge_data = (((u, v), G.adj[u][v]) for u, v in edges if G.has_edge(u, v))
    elif on_missing == 'default':
        edge_data = (((u, v), G.adj[u][v])
                     if G.has_edge(u, v) else ((u, v), {})
                     for u, v in edges)
    else:
        raise KeyError('on_missing={}'.format(on_missing))
    # Get `edge_attrs` desired value out of dictionary
    if on_keyerr == 'error':
        edge_attrs = ((e, d[key]) for e, d in edge_data)
    elif on_keyerr == 'filter':
        edge_attrs = ((e, d[key]) for e, d in edge_data if key in d)
    elif on_keyerr == 'default':
        edge_attrs = ((e, d.get(key, default)) for e, d in edge_data)
    else:
        raise KeyError('on_keyerr={}'.format(on_keyerr))
    return edge_attrs


def nx_gen_edge_values(G, key, edges=None, default=ub.NoParam,
                       on_missing='error', on_keyerr='default'):
    """
    Generates attributes values of specific edges

    Args:
        on_missing (str): Strategy for handling nodes missing from G.
            Can be {'error', 'default'}.  defaults to 'error'.
        on_keyerr (str): Strategy for handling keys missing from node dicts.
            Can be {'error', 'default'}.  defaults to 'default'
            if default is specified, otherwise defaults to 'error'.
    """
    if edges is None:
        edges = G.edges()
    if on_missing is None:
        on_missing = 'error'
    if on_keyerr is None:
        on_keyerr = 'default'
    if default is ub.NoParam and on_keyerr == 'default':
        on_keyerr = 'error'
    # Generate `data_iter` edges and data dictionary
    if on_missing == 'error':
        data_iter = (G.adj[u][v] for u, v in edges)
    elif on_missing == 'default':
        data_iter = (G.adj[u][v] if G.has_edge(u, v) else {}
                     for u, v in edges)
    else:
        raise KeyError('on_missing={} must be error, filter or default'.format(
            on_missing))
    # Get `value_iter` desired value out of dictionary
    if on_keyerr == 'error':
        value_iter = (d[key] for d in data_iter)
    elif on_keyerr == 'default':
        value_iter = (d.get(key, default) for d in data_iter)
    else:
        raise KeyError('on_keyerr={} must be error or default'.format(on_keyerr))
    return value_iter
    # if default is ub.NoParam:


def nx_edges(graph, keys=False, data=False):
    if graph.is_multigraph():
        edges = graph.edges(keys=keys, data=data)
    else:
        edges = graph.edges(data=data)
        #if keys:
        #    edges = [e[0:2] + (0,) + e[:2] for e in edges]
    return edges


def nx_delete_None_edge_attr(graph, edges=None):
    removed = 0
    if graph.is_multigraph():
        if edges is None:
            edges = list(graph.edges(keys=graph.is_multigraph()))
        for edge in edges:
            u, v, k = edge
            data = graph[u][v][k]
            for key in list(data.keys()):
                try:
                    if data[key] is None:
                        del data[key]
                        removed += 1
                except KeyError:
                    pass
    else:
        if edges is None:
            edges = list(graph.edges())
        for edge in graph.edges():
            u, v = edge
            data = graph[u][v]
            for key in list(data.keys()):
                try:
                    if data[key] is None:
                        del data[key]
                        removed += 1
                except KeyError:
                    pass
    return removed


def nx_delete_None_node_attr(graph, nodes=None):
    removed = 0
    if nodes is None:
        nodes = list(graph.nodes())
    for node in graph.nodes():
        node_dict = graph.nodes
        data = node_dict[node]
        for key in list(data.keys()):
            try:
                if data[key] is None:
                    del data[key]
                    removed += 1
            except KeyError:
                pass
    return removed


def nx_node_dict(G):
    print('Warning: use G.nodes instead')
    if nx.__version__.startswith('1'):
        return getattr(G, 'node')
    else:
        return G.nodes


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/graphid/graphid/util/nx_utils.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
