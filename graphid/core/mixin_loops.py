# -*- coding: utf-8 -*-
import numpy as np
import ubelt as ub
import pandas as pd
import itertools as it
from graphid.core import state as const
from graphid import util
from graphid.core.state import (POSTV, NEGTV, INCMP, NULL)
from graphid.core.refresh import RefreshCriteria


class InfrLoops(object):
    """
    Algorithm control flow loops
    """

    def main_gen(infr, max_loops=None, use_refresh=True):
        """
        The main outer loop.

        This function is designed as an iterator that will execute the graph
        algorithm main loop as automatically as possible, but if user input is
        needed, it will pause and yield the decision it needs help with. Once
        feedback is given for this item, you can continue the main loop by
        calling next. StopIteration is raised once the algorithm is complete.

        Args:
            max_loops(int): maximum number of times to run the outer loop,
                i.e. ranking is run at most this many times.
            use_refresh(bool): allow the refresh criterion to stop the algo

        Notes:
            Different phases of the main loop are implemented as subiterators

        CommandLine:
            python -m graphid.core.mixin_loops InfrLoops.main_gen

        Example:
            >>> from graphid.core.mixin_simulation import UserOracle
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(num_pccs=3, size=5)
            >>> infr.params['manual.n_peek'] = 10
            >>> infr.params['ranking.ntop'] = 1
            >>> infr.oracle = UserOracle(.99, rng=0)
            >>> infr.simulation_mode = False
            >>> infr.reset()
            >>> gen = infr.main_gen()
            >>> while True:
            >>>     try:
            >>>         reviews = next(gen)
            >>>         edge, priority, data = reviews[0]
            >>>         feedback = infr.request_oracle_review(edge)
            >>>         infr.add_feedback(edge, **feedback)
            >>>     except StopIteration:
            >>>         break
        """
        infr.print('Starting main loop', 1)
        infr.print('infr.params = {}'.format(ub.repr2(infr.params)))
        if max_loops is None:
            max_loops = infr.params['algo.max_outer_loops']
            if max_loops is None:
                max_loops = np.inf

        if infr.test_mode:
            print('------------------ {} -------------------'.format(infr.name))

        # Initialize a refresh criteria
        infr.init_refresh()

        # Phase 0.1: Ensure the user sees something immediately
        if infr.params['algo.quickstart']:
            infr.loop_phase = 'quickstart_init'
            # quick startup. Yield a bunch of random edges
            num = infr.params['manual.n_peek']
            user_request = []
            for edge in util.random_combinations(infr.aids, 2, num=num):
                user_request += [infr._make_review_tuple(edge, None)]
                yield user_request

        if infr.params['algo.hardcase']:
            infr.loop_phase = 'hardcase_init'
            # Check previously labeled edges that where the groundtruth and the
            # verifier disagree.
            yield from infr.hardcase_review_gen()

        if infr.params['inference.enabled']:
            infr.loop_phase = 'incon_recover_init'
            # First, fix any inconsistencies
            yield from infr.incon_recovery_gen()

        # Phase 0.2: Ensure positive redundancy (this is generally quick)
        # so the user starts seeing real work after one random review is made
        # unless the graph is already positive redundant.
        if infr.params['redun.enabled'] and infr.params['redun.enforce_pos']:
            infr.loop_phase = 'pos_redun_init'
            # Fix positive redundancy of anything within the loop
            yield from infr.pos_redun_gen()

        if infr.params['ranking.enabled']:
            for count in it.count(0):

                infr.print('Outer loop iter %d ' % (count,))

                # Phase 1: Try to merge PCCs by searching for LNBNN candidates
                infr.loop_phase = 'ranking_{}'.format(count)
                yield from infr.ranked_list_gen(use_refresh)

                terminate = (infr.refresh.num_meaningful == 0)
                if terminate:
                    infr.print('Triggered break criteria', 1, color='red')

                # Phase 2: Ensure positive redundancy.
                infr.loop_phase = 'posredun_{}'.format(count)
                if all(ub.take(infr.params, ['redun.enabled', 'redun.enforce_pos'])):
                    # Fix positive redundancy of anything within the loop
                    yield from infr.pos_redun_gen()

                print('prob_any_remain = %r' % (infr.refresh.prob_any_remain(),))
                print('infr.refresh.num_meaningful = {!r}'.format(
                    infr.refresh.num_meaningful))

                if (count + 1) >= max_loops:
                    infr.print('early stop', 1, color='red')
                    break

                if terminate:
                    infr.print('break triggered')
                    break

        if all(ub.take(infr.params, ['redun.enabled', 'redun.enforce_neg'])):
            # Phase 3: Try to automatically acheive negative redundancy without
            # asking the user to do anything but resolve inconsistency.
            infr.print('Entering phase 3', 1, color='red')
            infr.loop_phase = 'negredun'
            yield from infr.neg_redun_gen()

        infr.print('Terminate', 1, color='red')
        infr.print('Exiting main loop')

        if infr.params['inference.enabled']:
            infr.assert_consistency_invariant()

    def hardcase_review_gen(infr):
        """
        Subiterator for hardcase review

        Re-review non-confident edges that vsone did not classify correctly
        """
        infr.print('==============================', color='white')
        infr.print('--- HARDCASE PRIORITY LOOP ---', color='white')

        verifiers = infr.learn_evaluation_verifiers()
        verif = verifiers['match_state']

        edges_ = list(infr.edges())
        real_ = list(infr.edge_decision_from(edges_))
        flags_ = [r in {POSTV, NEGTV, INCMP} for r in real_]
        real = list(ub.compress(real_, flags_))
        edges = list(ub.compress(edges_, flags_))

        hardness = 1 - verif.easiness(edges, real)

        if True:
            df = pd.DataFrame({'edges': edges, 'real': real})
            df['hardness'] = hardness

            pred = verif.predict(edges)
            df['pred'] = pred.values

            df.sort_values('hardness', ascending=False)
            infr.print('hardness analysis')
            infr.print(str(df))

            infr.print('infr status: ' + ub.repr2(infr.status()))

        # Don't re-review anything that was confidently reviewed
        # CONFIDENCE = const.CONFIDENCE
        # CODE_TO_INT = CONFIDENCE.CODE_TO_INT.copy()
        # CODE_TO_INT[CONFIDENCE.CODE.UNKNOWN] = 0
        # conf = ub.take(CODE_TO_INT, infr.gen_edge_values(
        #     'confidence', edges, on_missing='default',
        #     default=CONFIDENCE.CODE.UNKNOWN))

        # This should only be run with certain params
        assert not infr.params['autoreview.enabled']
        assert not infr.params['redun.enabled']
        assert not infr.params['ranking.enabled']
        assert infr.params['inference.enabled']
        # const.CONFIDENCE.CODE.PRETTY_SURE
        if infr.params['queue.conf.thresh'] is None:
            # != 'pretty_sure':
            infr.print('WARNING: should queue.conf.thresh = "pretty_sure"?')

        # work around add_candidate_edges
        infr.prioritize(metric='hardness', edges=edges,
                        scores=hardness)
        infr.set_edge_attrs('hardness', ub.dzip(edges, hardness))
        yield from infr._inner_priority_gen(use_refresh=False)

    def ranked_list_gen(infr, use_refresh=True):
        """
        Subiterator for phase1 of the main algorithm

        Calls the underlying ranking algorithm and prioritizes the results
        """
        infr.print('============================', color='white')
        infr.print('--- RANKED LIST LOOP ---', color='white')
        n_prioritized = infr.refresh_candidate_edges()
        if n_prioritized == 0:
            infr.print('RANKING ALGO FOUND NO NEW EDGES')
            return
        if use_refresh:
            infr.refresh.clear()
        yield from infr._inner_priority_gen(use_refresh)

    def incon_recovery_gen(infr):
        """
        Subiterator for recovery mode of the mainm algorithm

        Iterates until the graph is consistent

        Note:
            inconsistency recovery is implicitly handled by the main algorithm,
            so other phases do not need to call this explicitly. This exists
            for the case where the only mode we wish to run is inconsistency
            recovery.
        """
        maybe_error_edges = list(infr.maybe_error_edges())
        if len(maybe_error_edges) == 0:
            raise StopIteration()
        infr.print('============================', color='white')
        infr.print('--- INCON RECOVER LOOP ---', color='white')
        infr.queue.clear()
        infr.add_candidate_edges(maybe_error_edges)
        yield from infr._inner_priority_gen(use_refresh=False)

    def pos_redun_gen(infr):
        """
        Subiterator for phase2 of the main algorithm.

        Searches for decisions that would commplete positive redundancy

        CommandLine:
            python -m graphid.core.mixin_loops InfrLoops.pos_redun_gen

        Example:
            >>> from graphid.core.mixin_loops import *
            >>> from graphid import demo
            >>> infr = demo.demodata_infr(num_pccs=3, size=5, pos_redun=1)
            >>> gen = infr.pos_redun_gen()
            >>> feedback = next(gen)
            >>> edge_ = feedback[0][0]
            >>> print(edge_)
            (1, 5)
        """
        infr.print('===========================', color='white')
        infr.print('--- POSITIVE REDUN LOOP ---', color='white')
        # FIXME: should prioritize inconsistentices first
        count = -1

        def serial_gen():
            # use this if threading does bad things
            if True:
                new_edges = list(infr.find_pos_redun_candidate_edges())
                if len(new_edges) > 0:
                    infr.add_candidate_edges(new_edges)
                    yield new_edges
            else:
                for new_edges in ub.chunks(infr.find_pos_redun_candidate_edges(), 100):
                    if len(new_edges) > 0:
                        infr.add_candidate_edges(new_edges)
                        yield new_edges

        def filtered_gen():
            # Buffer one-vs-one scores in the background and present an edge to
            # the user ASAP.
            # if infr.test_mode:
            candgen = serial_gen()
            for new_edges in candgen:
                yield new_edges

        for count in it.count(0):
            infr.print('check pos-redun iter {}'.format(count))
            infr.queue.clear()

            found_any = False

            for new_edges in filtered_gen():
                found_any = True
                gen = infr._inner_priority_gen(use_refresh=False)
                for value in gen:
                    yield value

            print('found_any = {!r}'.format(found_any))
            if not found_any:
                break

            infr.print('not pos-reduntant yet.', color='white')
        infr.print(
            'pos-redundancy achieved in {} iterations'.format(
                count + 1))

    def neg_redun_gen(infr):
        """
        Subiterator for phase3 of the main algorithm.

        Searches for decisions that would commplete negative redundancy
        """
        infr.print('===========================', color='white')
        infr.print('--- NEGATIVE REDUN LOOP ---', color='white')

        infr.queue.clear()

        only_auto = infr.params['redun.neg.only_auto']

        # TODO: outer loop that re-iterates until negative redundancy is
        # accomplished.
        needs_neg_redun = infr.find_neg_redun_candidate_edges()
        chunksize = 500
        for new_edges in ub.chunks(needs_neg_redun, chunksize):
            infr.print('another neg redun chunk')
            # Add chunks in a little at a time for faster response time
            infr.add_candidate_edges(new_edges)
            gen = infr._inner_priority_gen(use_refresh=False,
                                           only_auto=only_auto)
            for value in gen:
                yield value

    def _inner_priority_gen(infr, use_refresh=False, only_auto=False):
        """
        Helper function that implements the general inner priority loop.

        Executes reviews until the queue is empty or needs refresh

        Args:
            user_refresh (bool): if True enables the refresh criteria.
                (set to True in Phase 1)

            only_auto (bool) if True, then the user wont be prompted with
                reviews unless the graph is inconsistent.
                (set to True in Phase 3)

        Notes:
            The caller is responsible for populating the priority queue.  This
            will iterate until the queue is empty or the refresh critieron is
            triggered.
        """
        if infr.refresh:
            infr.refresh.enabled = use_refresh
        infr.print('Start inner loop with {} items in the queue'.format(
            len(infr.queue)))
        for count in it.count(0):
            if infr.is_recovering():
                infr.print('Still recovering after %d iterations' % (count,),
                           3, color='turquoise')
            else:
                # Do not check for refresh if we are recovering
                if use_refresh and infr.refresh.check():
                    infr.print('Triggered refresh criteria after %d iterations' %
                               (count,), 1, color='yellow')
                    break

            # If the queue is empty break
            if len(infr.queue) == 0:
                infr.print('No more edges after %d iterations, need refresh' %
                           (count,), 1, color='yellow')
                break

            # Try to automatically do the next review.
            edge, priority = infr.peek()
            infr.print('next_review. edge={}'.format(edge), 100)

            inconsistent = infr.is_recovering(edge)

            feedback = None
            if infr.params['autoreview.enabled'] and not inconsistent:
                # Try to autoreview if we aren't in an inconsistent state
                feedback = infr.try_auto_review(edge)

            if feedback is not None:
                # Add feedback from the automated method
                infr.add_feedback(edge, priority=priority, **feedback)
            else:
                # We can't automatically review, ask for help
                if only_auto and not inconsistent:
                    # We are in auto only mode, skip manual review
                    # unless there is an inconsistency
                    infr.skip(edge)
                else:
                    if infr.simulation_mode:
                        # Use oracle feedback
                        feedback = infr.request_oracle_review(edge)
                        infr.add_feedback(edge, priority=priority, **feedback)
                    else:
                        # Yield to the user if we need to pause
                        user_request = infr.emit_manual_review(edge, priority)
                        yield user_request

        if infr.metrics_list:
            infr._print_previous_loop_statistics(count)

    def init_refresh(infr):
        refresh_params = infr.subparams('refresh')
        infr.refresh = RefreshCriteria(**refresh_params)

    def start_id_review(infr, max_loops=None, use_refresh=None):
        assert infr._gen is None, 'algo already running'
        # Just exhaust the main generator
        infr._gen = infr.main_gen(max_loops=max_loops, use_refresh=use_refresh)
        # return infr._gen

    def main_loop(infr, max_loops=None, use_refresh=True):
        """ DEPRICATED

        use list(infr.main_gen) instead
        or assert not any(infr.main_gen())
        maybe this is fine.
        """
        infr.start_id_review(max_loops=max_loops, use_refresh=use_refresh)
        # To automatically run through the loop just exhaust the generator
        try:
            result = next(infr._gen)
            assert result is None, 'need user interaction. cannot auto loop'
        except StopIteration:
            pass
        infr._gen = None


class InfrReviewers(object):
    def try_auto_review(infr, edge):
        review = {
            'user_id': 'algo:auto_clf',
            'confidence': const.CONFIDENCE.CODE.PRETTY_SURE,
            'evidence_decision': None,
            'meta_decision': NULL,
            'timestamp_s1': None,
            'timestamp_c1': None,
            'timestamp_c2': None,
            'tags': [],
        }
        if infr.is_recovering():
            # Do not autoreview if we are in an inconsistent state
            infr.print('Must manually review inconsistent edge', 3)
            return None
        # Determine if anything passes the match threshold
        primary_task = 'match_state'

        try:
            decision_probs = infr.task_probs[primary_task][edge]
        except KeyError:
            if infr.verifiers is None:
                return None
            if infr.verifiers.get(primary_task, None) is None:
                return None
            # Compute probs if they haven't been done yet
            infr.ensure_priority_scores([edge])
            try:
                decision_probs = infr.task_probs[primary_task][edge]
            except KeyError:
                return None

        primary_thresh = infr.task_thresh[primary_task]
        decision_flags = {k: decision_probs[k] > thresh
                          for k, thresh in primary_thresh.items()}
        hasone = sum(decision_flags.values()) == 1
        auto_flag = False
        if hasone:
            try:
                # Check to see if it might be confounded by a photobomb
                pb_probs = infr.task_probs['photobomb_state'][edge]
                # pb_probs = infr.task_probs['photobomb_state'].loc[edge]
                # pb_probs = data['task_probs']['photobomb_state']
                pb_thresh = infr.task_thresh['photobomb_state']['pb']
                confounded = pb_probs['pb'] > pb_thresh
            except KeyError:
                print('Warning: confounding task probs not set (i.e. photobombs)')
                confounded = False
            if not confounded:
                # decision = decision_flags.argmax()
                evidence_decision = ub.argmax(decision_probs)
                review['evidence_decision'] = evidence_decision
                # truth = infr.match_state_gt(edge)
                truth = infr.dummy_verif._get_truth(edge)
                if review['evidence_decision'] != truth:
                    infr.print(
                        'AUTOMATIC ERROR edge={}, truth={}, decision={}, probs={}'.format(
                            edge, truth, review['evidence_decision'], decision_probs),
                        2, color='darkred')
                auto_flag = True
        if auto_flag and infr.verbose > 1:
            infr.print('Automatic review success')

        if auto_flag:
            return review
        else:
            return None

    def request_oracle_review(infr, edge, **kw):
        truth = infr.dummy_verif._get_truth(edge)
        # truth = infr.match_state_gt(edge)
        feedback = infr.oracle.review(edge, truth, infr, **kw)
        return feedback

    def _make_review_tuple(infr, edge, priority=None):
        """ Makes tuple to be sent back to the user """
        edge_data = infr.get_nonvisual_edge_data(
            edge, on_missing='default')
        # Extra information
        edge_data['nid_edge'] = infr.pos_graph.node_labels(*edge)
        if infr.queue is None:
            edge_data['queue_len'] = 0
        else:
            edge_data['queue_len'] = len(infr.queue)
        edge_data['n_ccs'] = (
            len(infr.pos_graph.connected_to(edge[0])),
            len(infr.pos_graph.connected_to(edge[1]))
        )
        return (edge, priority, edge_data)

    def emit_manual_review(infr, edge, priority=None):
        """
        Emits a signal containing edges that need review. The callback should
        present them to a user, get feedback, and then call on_accpet.
        """
        infr.print('emit_manual_review', 100)
        # Emit a list of reviews that can be considered.
        # The first is the most important
        user_request = []
        user_request += [infr._make_review_tuple(edge, priority)]
        try:
            for edge_, priority in infr.peek_many(infr.params['manual.n_peek']):
                if edge == edge_:
                    continue
                user_request += [infr._make_review_tuple(edge_, priority)]
        except TypeError:
            pass

        # If registered, send the request via a callback.
        request_review = infr.callbacks.get('request_review', None)
        if request_review is not None:
            # Send these reviews to a user
            request_review(user_request)
        # Otherwise the current process must handle the request by return value
        return user_request

    def skip(infr, edge):
        infr.print('skipping edge={}'.format(edge), 100)
        try:
            del infr.queue[edge]
        except Exception:
            pass

    def accept(infr, feedback):
        """
        Called when user has completed feedback from qt or web
        """
        annot1_state = feedback.pop('annot1_state', None)
        annot2_state = feedback.pop('annot2_state', None)
        if annot1_state:
            infr.add_node_feedback(**annot1_state)
        if annot2_state:
            infr.add_node_feedback(**annot2_state)
        infr.add_feedback(**feedback)

    def continue_review(infr):
        infr.print('continue_review', 10)
        if infr._gen is None:
            return None
        try:
            user_request = next(infr._gen)
        except StopIteration:
            review_finished = infr.callbacks.get('review_finished', None)
            if review_finished is not None:
                review_finished()
            infr._gen = None
            user_request = None
        return user_request

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/graphid/graphid.core/mixin_loops.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
