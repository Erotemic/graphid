"""
Contains the abstract definitions of the ranking and verification algorithms
These must be overloaded and defined depending on the end application.

Example:
    >>> from graphid import demo
    >>> from graphid.core import abstract
    >>> kwargs = dict(num_pccs=40, size=2)
    >>> infr = demo.demodata_infr(**kwargs)
    >>> class CustomRanker(abstract.Ranker):
    >>>     def __init__(ranker, infr):
    >>>         # Allow the custom ranker to know about the inference object
    >>>         ranker.infr = infr
    >>>         ranker.times_called = 0
    >>>     def predict_single_ranking(ranker, node, K=10):
    >>>         # Just return a random ranking
    >>>         ranker.times_called += 1
    >>>         others = set(ranker.infr.graph.nodes()) - {node}
    >>>         ranked_nodes = list(others)[0:K]
    >>>         return ranked_nodes
    >>> # Create and register the ranking algorithm
    >>> ranker = CustomRanker(infr)
    >>> infr.set_ranker(ranker)
    >>> # Can now use the ranker by itself
    >>> ranked_nodes = infr.ranker.predict_single_ranking(10)
    >>> assert ranker.times_called == 1
    >>> # The ranker will be used by refreshing candidate edges
    >>> infr.refresh_candidate_edges()
    >>> assert ranker.times_called > 1
"""
import abc
from graphid import util


class Ranker(abc.ABC):
    """
    The abstract API that a ranking algorithm must implement.

    The ranker must be aware of the current database, so its often a good idea
    to construct the ranker such that the `infr` object is given in its
    constructor.
    """

    @abc.abstractmethod
    def predict_single_ranking(ranker, node, K=10):
        """
        Rank a single query

        Abstract method to define how to query a single annotation.
        Either this or `predict_rankings` must be implemented.

        Args:
            node (int): a query node
            K (int): number of nearest neighbors

        Returns:
            list: ranked_nodes: a list of ranked nodes
        """
        raise NotImplementedError('abstract method, must overwrite')

    def predict_rankings(ranker, nodes, K=10):
        """
        Yields a list ranked edges connected to each node.

        Abstract method to define how to query a multiple annotations. Either
        this or `predict_single_ranking` must be implemented.

        Args:
            nodes (list): a sequence of query node
            K (int): number of nearest neighbors

        Returns:
            list: a list of ranked nodes
        """
        for u in nodes:
            yield ranker.predict_single_ranking(u, K=K)

    def predict_candidate_edges(ranker, nodes, K=10):
        """
        Uses `predict_rankings` to construct a set of candidate edges.

        These edges may already exist in the graph, in which case the caller
        (i.e. the AnnotInference object) must handle duplicates.

        Returns:
            set: new_edges: set of tuples indicating edges in the graph
        """
        new_edges = []
        for u, ranks in zip(nodes, ranker.predict_rankings(nodes, K=K)):
            new_edges.extend([util.e_(u, v) for v in ranks])
        new_edges = set(new_edges)
        return new_edges


class Verifier(object):
    """
    The abstract API that a verification algorithm must implement.
    """

    @abc.abstractmethod
    def predict_proba_df(verif, edges):
        """
        Returns classification probabilities given a set of annotation pairs
        for a particular task (e.g. matching / photobomb)

        Args:
            edges (list): list of (u, v) edges (i.e. annotation pairs)

        Returns:
            pd.DataFrame: data frame where the index is a multindex of keys
                aid1, aid2 representing the input edges. The columns must be
                the possible classifications (e.g. positive, negative,
                incomparable) of each edge, and each row must sum to one.

        Notes:
            Use the following snippet to make the correct DataFame structure
            probs = pd.DataFrame(
                list(probs_of_each_edge)
                index=util.ensure_multi_index(edges, ('aid1', 'aid2'))
            )
        """
        raise NotImplementedError('must implement')
