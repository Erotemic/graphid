"""
Functionality related to the k-edge redundancy measures
"""
import ubelt as ub
import numpy as np
import networkx as nx
import itertools as it
from graphid.util import nx_utils as nxu
from graphid import util
from graphid.util.nx_utils import e_
from graphid.core.state import (POSTV, NEGTV, INCMP, UNREV, UNKWN, UNINFERABLE)  # NOQA
from graphid.core.state import (SAME, DIFF, NULL)  # NOQA


class _RedundancyComputers:
    """
    methods for computing redundancy

    These are used to compute redundancy bookkeeping structures.
    Thus, they should not use them in their calculations.
    """

    # def pos_redundancy(infr, cc):
    #     """ Returns how positive redundant a cc is """
    #     pos_subgraph = infr.pos_graph.subgraph(cc, dynamic=False)
    #     if nxu.is_complete(pos_subgraph):
    #         return np.inf
    #     else:
    #         return nx.edge_connectivity(pos_subgraph)

    # def neg_redundancy(infr, cc1, cc2):
    #     """ Returns how negative redundant a cc is """
    #     neg_edge_gen = nxu.edges_cross(infr.neg_graph, cc1, cc2)
    #     num_neg = len(list(neg_edge_gen))
    #     if num_neg == len(cc1) or num_neg == len(cc2):
    #         return np.inf
    #     else:
    #         return num_neg

    def is_pos_redundant(infr, cc, k=None, relax=None, assume_connected=False):
        """
        Tests if a group of nodes is positive redundant.
        (ie. if the group is k-edge-connected)

        CommandLine:
            python -m graphid.core.mixin_dynamic _RedundancyComputers.is_pos_redundant

        Example:
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(ccs=[(1, 2, 3)], pos_redun=1)
            >>> cc = infr.pos_graph.connected_to(1)
            >>> flag1 = infr.is_pos_redundant(cc)
            >>> infr.add_feedback((1, 3), POSTV)
            >>> flag2 = infr.is_pos_redundant(cc, k=2)
            >>> flags = [flag1, flag2]
            >>> print('flags = %r' % (flags,))
            flags = [False, True]
            >>> # xdoc: +REQUIRES(--show)
            >>> from graphid import util
            >>> infr.show()
            >>> util.show_if_requested()
        """
        if k is None:
            k = infr.params['redun.pos']
        if assume_connected and k == 1:
            return True  # assumes cc is connected
        if relax is None:
            relax = True
        pos_subgraph = infr.pos_graph.subgraph(cc, dynamic=False)
        if relax:
            # If we cannot add any more edges to the subgraph then we consider
            # it positive redundant.
            n_incomp = sum(1 for _ in nxu.edges_inside(infr.incomp_graph, cc))
            n_pos = pos_subgraph.number_of_edges()
            n_nodes = pos_subgraph.number_of_nodes()
            n_max = (n_nodes * (n_nodes - 1)) // 2
            if n_max == (n_pos + n_incomp):
                return True
        # In all other cases test edge-connectivity
        return nxu.is_k_edge_connected(pos_subgraph, k=k)

    def is_neg_redundant(infr, cc1, cc2, k=None):
        """
        Tests if two disjoint groups of nodes are negative redundant
        (ie. have at least k negative edges between them).

        CommandLine:
            python -m graphid.core.mixin_dynamic _RedundancyComputers.is_neg_redundant --show

        Example:
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(ccs=[(1, 2), (3, 4)], ignore_pair=True)
            >>> infr.params['redun.neg'] = 2
            >>> cc1 = infr.pos_graph.connected_to(1)
            >>> cc2 = infr.pos_graph.connected_to(3)
            >>> flag1 = infr.is_neg_redundant(cc1, cc2)
            >>> infr.add_feedback((1, 3), NEGTV)
            >>> flag2 = infr.is_neg_redundant(cc1, cc2)
            >>> infr.add_feedback((2, 4), NEGTV)
            >>> flag3 = infr.is_neg_redundant(cc1, cc2)
            >>> flags = [flag1, flag2, flag3]
            >>> print('flags = %r' % (flags,))
            >>> assert flags == [False, False, True]
            >>> # xdoc: +REQUIRES(--show)
            >>> from graphid import util
            >>> infr.show()
            >>> util.show_if_requested()
        """
        if k is None:
            k = infr.params['redun.neg']
        neg_edge_gen = nxu.edges_cross(infr.neg_graph, cc1, cc2)
        # do a lazy count of negative edges
        for count, _ in enumerate(neg_edge_gen, start=1):
            if count >= k:
                return True
        return False

    def find_neg_nids_to(infr, cc):
        """
        Find the nids with at least one negative edge external
        to this cc.
        """
        pos_graph = infr.pos_graph
        neg_graph = infr.neg_graph
        out_neg_nids = set([])
        for u in cc:
            nid1 = pos_graph.node_label(u)
            for v in neg_graph.neighbors(u):
                nid2 = pos_graph.node_label(v)
                if nid1 == nid2 and v not in cc:
                    continue
                out_neg_nids.add(nid2)
        return out_neg_nids

    def find_neg_nid_freq_to(infr, cc):
        """
        Find the number of edges leaving `cc` and directed towards specific
        names.
        """
        pos_graph = infr.pos_graph
        neg_graph = infr.neg_graph
        neg_nid_freq = ub.ddict(lambda: 0)
        for u in cc:
            nid1 = pos_graph.node_label(u)
            for v in neg_graph.neighbors(u):
                nid2 = pos_graph.node_label(v)
                if nid1 == nid2 and v not in cc:
                    continue
                neg_nid_freq[nid2] += 1
        return neg_nid_freq

    def find_neg_redun_nids_to(infr, cc):
        """
        Get PCCs that are k-negative redundant with `cc`

        Example:
            >>> from graphid import demo
            >>> infr = demo.demodata_infr()
            >>> node = 20
            >>> cc = infr.pos_graph.connected_to(node)
            >>> infr.params['redun.neg'] = 2
            >>> infr.find_neg_redun_nids_to(cc)
        """
        neg_nid_freq = infr.find_neg_nid_freq_to(cc)
        # check for k-negative redundancy
        k_neg = infr.params['redun.neg']
        pos_graph = infr.pos_graph
        neg_nids = [
            nid2 for nid2, freq in neg_nid_freq.items()
            if (
                freq >= k_neg or
                freq == len(cc) or
                freq == len(pos_graph.connected_to(nid2))
            )
        ]
        return neg_nids

    def find_pos_redundant_pccs(infr, k=None, relax=None):
        if k is None:
            k = infr.params['redun.pos']
        for cc in infr.consistent_components():
            if infr.is_pos_redundant(cc, k=k, relax=relax):
                yield cc

    def find_non_pos_redundant_pccs(infr, k=None, relax=None):
        """
        Get PCCs that are not k-positive-redundant
        """
        if k is None:
            k = infr.params['redun.pos']
        for cc in infr.consistent_components():
            if not infr.is_pos_redundant(cc, k=k, relax=relax):
                yield cc

    def find_non_neg_redun_pccs(infr, k=None, cc=None):
        """
        Get pairs of PCCs that are not complete.

        Args:
            k (int): level of redunency to be considered complete
            cc (set, optional): if specified only search for
                other pccs that are not negative redundant to
                this particular cc

        Example:
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(pcc_sizes=[1, 1, 2, 3, 5, 8], ignore_pair=True)
            >>> non_neg_pccs = list(infr.find_non_neg_redun_pccs(k=2))
            >>> assert len(non_neg_pccs) == (6 * 5) / 2
        """
        if k is None:
            k = infr.params['redun.neg']
        # need to ensure pccs is static in case new user input is added
        pccs = list(infr.positive_components())
        if cc is None:
            # Loop through all pairs
            for cc1, cc2 in it.combinations(pccs, 2):
                if not infr.is_neg_redundant(cc1, cc2, k=k):
                    yield cc1, cc2
        else:
            cc1 = cc
            for cc2 in pccs:
                if cc1 != cc2:
                    if not infr.is_neg_redundant(cc1, cc2, k=k):
                        yield cc1, cc2

    def find_pos_redun_nids(infr):
        """ recomputes infr.pos_redun_nids """
        for cc in infr.find_pos_redundant_pccs():
            node = next(iter(cc))
            nid = infr.pos_graph.node_label(node)
            yield nid

    def find_neg_redun_nids(infr):
        """ recomputes edges in infr.neg_redun_metagraph """
        for cc in infr.consistent_components():
            node = next(iter(cc))
            nid1 = infr.pos_graph.node_label(node)
            for nid2 in infr.find_neg_redun_nids_to(cc):
                if nid1 < nid2:
                    yield nid1, nid2


class _RedundancyAugmentation:

    def find_neg_augment_edges(infr, cc1, cc2, k=None):
        """
        Find enough edges to between two pccs to make them k-negative complete
        The two CCs should be disjoint and not have any positive edges between
        them.

        Args:
            cc1 (set): nodes in one PCC
            cc2 (set): nodes in another positive-disjoint PCC
            k (int): redundnacy level (if None uses infr.params['redun.neg'])

        Example:
            >>> from graphid import demo
            >>> k = 2
            >>> cc1, cc2 = {1}, {2, 3}
            >>> # --- return an augmentation if feasible
            >>> infr = demo.demodata_infr(ccs=[cc1, cc2], ignore_pair=True)
            >>> edges = set(infr.find_neg_augment_edges(cc1, cc2, k=k))
            >>> assert edges == {(1, 2), (1, 3)}
            >>> # --- if infeasible return a partial augmentation
            >>> infr.add_feedback((1, 2), INCMP)
            >>> edges = set(infr.find_neg_augment_edges(cc1, cc2, k=k))
            >>> assert edges == {(1, 3)}
        """
        if k is None:
            k = infr.params['redun.neg']
        assert cc1 is not cc2, 'CCs should be disjoint (but they are the same)'
        assert len(cc1.intersection(cc2)) == 0, 'CCs should be disjoint'
        existing_edges = set(nxu.edges_cross(infr.graph, cc1, cc2))

        reviewed_edges = {
            edge: state
            for edge, state in zip(existing_edges,
                                   infr.edge_decision_from(existing_edges))
            if state != UNREV
        }

        # Find how many negative edges we already have
        num = sum([state == NEGTV for state in reviewed_edges.values()])
        if num < k:
            # Find k random negative edges
            check_edges = existing_edges - set(reviewed_edges)
            # Check the existing but unreviewed edges first
            for edge in check_edges:
                num += 1
                yield edge
                if num >= k:
                    return
                    # raise StopIteration()
            # Check non-existing edges next
            seed = 2827295125
            try:
                seed += sum(cc1) + sum(cc2)
            except Exception:
                pass
            rng = np.random.RandomState(seed)
            cc1 = util.shuffle(list(cc1), rng=rng)
            cc2 = util.shuffle(list(cc2), rng=rng)
            cc1 = util.shuffle(list(cc1), rng=rng)
            for edge in it.starmap(nxu.e_, nxu.diag_product(cc1, cc2)):
                if edge not in existing_edges:
                    num += 1
                    yield edge
                    if num >= k:
                        return
                        # raise StopIteration()

    def find_pos_augment_edges(infr, pcc, k=None):
        """
        # [[1, 0], [0, 2], [1, 2], [3, 1]]
        pos_sub = nx.Graph([[0, 1], [1, 2], [0, 2], [1, 3]])
        """
        if k is None:
            pos_k = infr.params['redun.pos']
        else:
            pos_k = k
        pos_sub = infr.pos_graph.subgraph(pcc)

        # TODO:
        # weight by pairs most likely to be comparable

        # First try to augment only with unreviewed existing edges
        unrev_avail = list(nxu.edges_inside(infr.unreviewed_graph, pcc))
        try:
            check_edges = list(nxu.k_edge_augmentation(
                pos_sub, k=pos_k, avail=unrev_avail, partial=False))
        except nx.NetworkXUnfeasible:
            check_edges = None
        if not check_edges:
            # Allow new edges to be introduced
            full_sub = infr.graph.subgraph(pcc).copy()
            new_avail = util.estarmap(infr.e_, nx.complement(full_sub).edges())
            full_avail = unrev_avail + new_avail
            n_max = (len(pos_sub) * (len(pos_sub) - 1)) // 2
            n_complement = n_max - pos_sub.number_of_edges()
            if len(full_avail) == n_complement:
                # can use the faster algorithm
                check_edges = list(nxu.k_edge_augmentation(
                    pos_sub, k=pos_k, partial=True))
            else:
                # have to use the slow approximate algo
                check_edges = list(nxu.k_edge_augmentation(
                    pos_sub, k=pos_k, avail=full_avail, partial=True))
        check_edges = set(it.starmap(e_, check_edges))
        return check_edges

    def find_pos_redun_candidate_edges(infr, k=None, verbose=False):
        """
        Searches for augmenting edges that would make PCCs k-positive redundant

        CommandLine:
            python -m graphid.core.mixin_dynamic _RedundancyAugmentation.find_pos_redun_candidate_edges

        Doctest:
            >>> from graphid.core.mixin_redundancy import *  # NOQA
            >>> from graphid import demo
            >>> # FIXME: this behavior seems to change depending on Python version
            >>> infr = demo.demodata_infr(ccs=[(1, 2, 3, 4, 5), (7, 8, 9, 10)], pos_redun=1)
            >>> infr.add_feedback((2, 5), POSTV)
            >>> infr.add_feedback((1, 5), INCMP)
            >>> infr.params['redun.pos'] = 2
            >>> candidate_edges = sorted(infr.find_pos_redun_candidate_edges())
            ...
            >>> result = ('candidate_edges = ' + ub.urepr(candidate_edges, nl=0))
            >>> print(result)
            candidate_edges = [(1, 4), ..., (7, 10)]

        Ignore:
            print(nx.write_network_text(infr.neg_graph))
            print(nx.write_network_text(infr.pos_graph))
            print(nx.write_network_text(infr.incomp_graph))
        """
        # Add random edges between exisiting non-redundant PCCs
        if k is None:
            k = infr.params['redun.pos']
        # infr.find_non_pos_redundant_pccs(k=k, relax=True)
        pcc_gen = list(infr.positive_components())
        prog = ub.ProgIter(pcc_gen, enabled=verbose, freq=1, adjust=False)
        for pcc in prog:
            if not infr.is_pos_redundant(pcc, k=k, relax=True,
                                         assume_connected=True):
                for edge in infr.find_pos_augment_edges(pcc, k=k):
                    yield nxu.e_(*edge)

    def find_neg_redun_candidate_edges(infr, k=None):
        """
        Get pairs of PCCs that are not complete.
        Finds edges that might complete them.

        Example:
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(ccs=[(1,), (2,), (3,)], ignore_pair=True)
            >>> edges = list(infr.find_neg_redun_candidate_edges())
            >>> assert len(edges) == 3, 'all should be needed here'
            >>> infr.add_feedback_from(edges, evidence_decision=NEGTV)
            >>> assert len(list(infr.find_neg_redun_candidate_edges())) == 0

        Example:
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(pcc_sizes=[3] * 20, ignore_pair=True)
            >>> ccs = list(infr.positive_components())
            >>> gen = infr.find_neg_redun_candidate_edges(k=2)
            >>> for edge in gen:
            >>>     # What happens when we make ccs positive
            >>>     print(infr.node_labels(edge))
            >>>     infr.add_feedback(edge, evidence_decision=POSTV)
            >>> import ubelt as ub
            >>> infr = demo.demodata_infr(pcc_sizes=[1] * 30, ignore_pair=True)
            >>> ccs = list(infr.positive_components())
            >>> gen = infr.find_neg_redun_candidate_edges(k=3)
            >>> for chunk in ub.chunks(gen, 2):
            >>>     for edge in chunk:
            >>>         # What happens when we make ccs positive
            >>>         print(infr.node_labels(edge))
            >>>         infr.add_feedback(edge, evidence_decision=POSTV)

            list(gen)
        """
        if k is None:
            k = infr.params['redun.neg']
        # Loop through all pairs
        for cc1, cc2 in infr.find_non_neg_redun_pccs(k=k):
            if len(cc1.intersection(cc2)) > 0:
                # If there is modification of the underlying graph while we
                # iterate, then two ccs may not be disjoint. Skip these cases.
                continue
            for u, v in infr.find_neg_augment_edges(cc1, cc2, k):
                edge = e_(u, v)
                infr.assert_edge(edge)
                yield edge


class Redundancy(_RedundancyComputers, _RedundancyAugmentation):
    """ methods for dynamic redundancy book-keeping """

    # def pos_redun_edge_flag(infr, edge):
    #     """ Quickly check if edge is flagged as pos redundant """
    #     nid1, nid2 = infr.pos_graph.node_labels(*edge)
    #     return nid1 == nid2 and nid1 in infr.pos_redun_nids

    # def neg_redun_edge_flag(infr, edge):
    #     """ Quickly check if edge is flagged as neg redundant """
    #     nid1, nid2 = infr.pos_graph.node_labels(*edge)
    #     return infr.neg_redun_metagraph.has_edge(nid1, nid2)

    def is_flagged_as_redun(infr, edge):
        """
        Tests redundancy against bookkeeping structure against cache
        """
        nidu, nidv = infr.node_labels(*edge)
        if nidu == nidv:
            if nidu in infr.pos_redun_nids:
                return True
        elif nidu != nidv:
            if infr.neg_redun_metagraph.has_edge(nidu, nidv):
                return True
        return False

    def filter_edges_flagged_as_redun(infr, edges):
        """
        Returns only edges that are not flagged as redundant.
        Uses bookkeeping structures

        Example:
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(num_pccs=1, size=4)
            >>> infr.clear_edges()
            >>> infr.ensure_cliques()
            >>> infr.clear_feedback()
            >>> print(ub.urepr(infr.status()))
            >>> nonredun_edges = list(infr.filter_edges_flagged_as_redun(
            >>>     infr.unreviewed_graph.edges()))
            >>> assert len(nonredun_edges) == 6
        """
        for edge in edges:
            if not infr.is_flagged_as_redun(edge):
                yield edge

    def update_extern_neg_redun(infr, nid, may_add=True, may_remove=True,
                                force=False):
        """
        Checks if `nid` is negative redundant to any other `cc` it has at least
        one negative review to.
        (TODO: NEG REDUN CAN BE CONSOLIDATED VIA NEG-META-GRAPH)
        """
        if not infr.params['redun.enabled']:
            return
        # infr.print('neg_redun external update nid={}'.format(nid), 5)
        k_neg = infr.params['redun.neg']
        cc1 = infr.pos_graph.component(nid)
        force = True
        if force:
            # TODO: non-force versions
            freqs = infr.find_neg_nid_freq_to(cc1)
            other_nids = []
            flags = []
            for other_nid, freq in freqs.items():
                if freq >= k_neg:
                    other_nids.append(other_nid)
                    flags.append(True)
                elif may_remove:
                    other_nids.append(other_nid)
                    flags.append(False)

            if len(other_nids) > 0:
                infr._set_neg_redun_flags(nid, other_nids, flags)
            else:
                infr.print('neg_redun skip update nid=%r' % (nid,), 6)

    def update_neg_redun_to(infr, nid1, other_nids, may_add=True, may_remove=True,
                            force=False):
        """
        Checks if nid1 is neg redundant to other_nids.
        Edges are either removed or added to the queue appropriately.
        (TODO: NEG REDUN CAN BE CONSOLIDATED VIA NEG-META-GRAPH)
        """
        if not infr.params['redun.enabled']:
            return
        # infr.print('update_neg_redun_to', 5)
        force = True
        cc1 = infr.pos_graph.component(nid1)
        if not force:
            raise NotImplementedError('implement non-forced version')
        flags = []
        for nid2 in other_nids:
            cc2 = infr.pos_graph.component(nid2)
            need_add = infr.is_neg_redundant(cc1, cc2)
            flags.append(need_add)
        infr._set_neg_redun_flags(nid1, other_nids, flags)

    def update_pos_redun(infr, nid, may_add=True, may_remove=True,
                         force=False):
        """
        Checks if a PCC is newly, or no longer positive redundant.
        Edges are either removed or added to the queue appropriately.
        """
        if not infr.params['redun.enabled']:
            return

        # force = True
        # infr.print('update_pos_redun')
        need_add = False
        need_remove = False
        if force:
            cc = infr.pos_graph.component(nid)
            need_add = infr.is_pos_redundant(cc)
            need_remove = not need_add
        else:
            was_pos_redun = nid in infr.pos_redun_nids
            if may_add and not was_pos_redun:
                cc = infr.pos_graph.component(nid)
                need_add = infr.is_pos_redundant(cc)
            elif may_remove and not was_pos_redun:
                cc = infr.pos_graph.component(nid)
                need_remove = not infr.is_pos_redundant(cc)
        if need_add:
            infr._set_pos_redun_flag(nid, True)
        elif need_remove:
            infr._set_pos_redun_flag(nid, False)
        else:
            infr.print('pos_redun skip update nid=%r' % (nid,), 6)

    def _set_pos_redun_flag(infr, nid, flag):
        """
        Flags or unflags an nid as positive redundant.
        """
        was_pos_redun = nid in infr.pos_redun_nids
        if flag:
            if not was_pos_redun:
                infr.print('pos_redun flag=T nid=%r' % (nid,), 5)
            else:
                infr.print('pos_redun flag=T nid=%r (already done)' % (nid,), 6)
            infr.pos_redun_nids.add(nid)
            cc = infr.pos_graph.component(nid)
            infr.remove_internal_priority(cc)
            if infr.params['inference.update_attrs']:
                infr.set_edge_attrs(
                    'inferred_state',
                    ub.dzip(nxu.edges_inside(infr.graph, cc), ['same'])
                )
        else:
            if was_pos_redun:
                infr.print('pos_redun flag=F nid=%r' % (nid,), 5)
            else:
                infr.print('pos_redun flag=F nid=%r (already done)' % (nid,), 6)
            cc = infr.pos_graph.component(nid)
            infr.pos_redun_nids -= {nid}
            infr.reinstate_internal_priority(cc)
            if infr.params['inference.update_attrs']:
                infr.set_edge_attrs(
                    'inferred_state',
                    ub.dzip(nxu.edges_inside(infr.graph, cc), [None])
                )

    def _set_neg_redun_flags(infr, nid1, other_nids, flags):
        """
        Flags or unflags an nid1 as negative redundant with other nids.
        (TODO: NEG REDUN CAN BE CONSOLIDATED VIA NEG-META-GRAPH)
        """
        needs_unflag = []
        needs_flag = []
        already_flagged = []
        already_unflagged = []
        cc1 = infr.pos_graph.component(nid1)
        other_nids = list(other_nids)

        # Determine what needs what
        for nid2, flag in zip(other_nids, flags):
            was_neg_redun = infr.neg_redun_metagraph.has_edge(nid1, nid2)
            if flag:
                if not was_neg_redun:
                    needs_flag.append(nid2)
                else:
                    already_flagged.append(nid2)
            else:
                if was_neg_redun:
                    needs_unflag.append(nid2)
                else:
                    already_unflagged.append(nid2)

        # Print summary of what will be done
        def _print_helper(what, others, already=False):
            if len(others) == 0:
                return
            n_other_thresh = 4
            if len(others) > n_other_thresh:
                omsg = '#others={}'.format(len(others))
            else:
                omsg = 'others={}'.format(others)
            amsg = '(already done)' if already else ''
            msg = '{} nid={}, {} {}'.format(what, nid1, omsg, amsg)
            infr.print(msg, 5 + already)

        _print_helper('neg_redun flag=T', needs_flag)
        _print_helper('neg_redun flag=T', already_flagged, already=True)
        _print_helper('neg_redun flag=F', needs_unflag)
        _print_helper('neg_redun flag=F', already_unflagged, already=True)

        # Do the flagging/unflagging
        for nid2 in needs_flag:
            infr.neg_redun_metagraph.add_edge(nid1, nid2)
        for nid2 in needs_unflag:
            infr.neg_redun_metagraph.remove_edge(nid1, nid2)

        # Update priorities and attributes
        if infr.params['inference.update_attrs'] or infr.queue is not None:
            all_flagged_edges = []
            # Unprioritize all edges between flagged nids
            for nid2 in it.chain(needs_flag, already_flagged):
                cc2 = infr.pos_graph.component(nid2)
                all_flagged_edges.extend(nxu.edges_cross(infr.graph, cc1, cc2))

        if infr.queue is not None or infr.params['inference.update_attrs']:
            all_unflagged_edges = []
            unrev_unflagged_edges = []
            unrev_graph = infr.unreviewed_graph
            # Reprioritize unreviewed edges between unflagged nids
            # Marked inferred state of all edges
            for nid2 in it.chain(needs_unflag, already_unflagged):
                cc2 = infr.pos_graph.component(nid2)
                if infr.queue is not None:
                    _edges = nxu.edges_cross(unrev_graph, cc1, cc2)
                    unrev_unflagged_edges.extend(_edges)
                if infr.params['inference.update_attrs']:
                    _edges = nxu.edges_cross(infr.graph, cc1, cc2)
                    all_unflagged_edges.extend(_edges)

            # Batch set prioritize
            infr._remove_edge_priority(all_flagged_edges)
            infr._reinstate_edge_priority(unrev_unflagged_edges)

            if infr.params['inference.update_attrs']:
                infr.set_edge_attrs(
                    'inferred_state', ub.dzip(all_flagged_edges, ['diff'])
                )
                infr.set_edge_attrs(
                    'inferred_state', ub.dzip(all_unflagged_edges, [None])
                )

    def _purge_redun_flags(infr, nid):
        """
        Removes positive and negative redundancy from nids and all other PCCs
        touching nids respectively. Return the external PCC nids.

        (TODO: NEG REDUN CAN BE CONSOLIDATED VIA NEG-META-GRAPH)
        """
        if not infr.params['redun.enabled']:
            return []
        if infr.neg_redun_metagraph.has_node(nid):
            prev_neg_nids = set(infr.neg_redun_metagraph.neighbors(nid))
        else:
            prev_neg_nids = []
        # infr.print('_purge, nid=%r, prev_neg_nids = %r' % (nid, prev_neg_nids,))
        # for other_nid in prev_neg_nids:
        #     flag = False
        #     if other_nid not in infr.pos_graph._ccs:
        #         flag = True
        #         infr.print('!!nid=%r did not update' % (other_nid,))
        #     if flag:
        #         assert flag, 'nids not maintained'
        for other_nid in prev_neg_nids:
            infr._set_neg_redun_flags(nid, [other_nid], [False])
        if nid in infr.pos_redun_nids:
            infr._set_pos_redun_flag(nid, False)
        return prev_neg_nids
