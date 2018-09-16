import ubelt as ub
from graphid import util
from graphid.core import annot_inference


class GraphID(ub.NiceRepr):
    """
    Public API for the Graph ID algorithm

    Example:
        >>> # DISABLE_DOCTEST
        >>> for query in iter(self):
        >>>     feedback = oracle.review(query)
        >>>     self.add_feedback(feedback)
    """

    def __init__(self):
        self.infr = None
        self.infr = annot_inference.AnnotInference()

    def add_annots_from(self, annots):
        self.infr.add_aids(annots)

    def add_edges_from(self, edges):
        self.infr.add_feedback_from(edges)

    def add_edge(self, edge, evidence_decision=None):
        self.infr.add_feedback(edge, evidence_decision=evidence_decision)

    def __iter__(self):
        """
        Defines an iterable interface that iterates through the priority queue
        until there are no more edges to review. Note: if reviews dont happen
        between calls to next this algorithm will infinitely loop. It must be
        used interactively with an external agent.
        """
        while True:
            yield self.peek()

    def peek(self, n=0):
        """
        Look at the next `n` items in the priority queue.
        When n=0 we only return one item, otherwise we return a list of items.
        (Note: We only make gaurentees about the first)
        """
        return self.infr.peek(n)

    def subgraph(self, aids):
        new_self = self.__class__()
        new_self.infr = self.infr.subgraph(aids)
        return new_self

    def pccs(self):
        """
        Positive Connected Components

        Yeilds:
            list: list of aids indicating all annotations currently predicted
                to be some specific individual / category.

            list : list of aids : Current prediction of individuals.
        """
        yield from self.infr.pccs

    def is_consistent(self):
        """
        Returns:
            bool: if any PCC contains a
        """
        return self.infr.is_consistent

    def add_feedback(self, edge, **kwargs):
        """
        Adds the information from a review to the graph for consideration in
        the dynamic inference algorithm.
        """
        kw = util.KWSpec(evidence_decision=None, tags=None, user_id=None,
                         meta_decision=None, confidence=None,
                         timestamp_c1=None, timestamp_c2=None,
                         timestamp_s1=None, timestamp=None, verbose=None,
                         priority=None)(**kwargs)
        kw
