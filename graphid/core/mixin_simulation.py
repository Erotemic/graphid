"""
Mixin functionality for experiments, tests, and simulations.
This includes recordings measures used to generate plots in JC's thesis.
"""
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import itertools as it
import ubelt as ub
import pandas as pd
from functools import partial
from graphid.core.state import (POSTV, NEGTV, INCMP, UNREV, UNKWN, NULL)
from graphid import util


class SimulationHelpers(object):
    def init_simulation(infr, oracle_accuracy=1.0, k_redun=2,
                        enable_autoreview=True, enable_inference=True,
                        classifiers=None, match_state_thresh=None,
                        max_outer_loops=None, name=None):
        infr.print('INIT SIMULATION', color='yellow')

        infr.name = name
        infr.simulation_mode = True

        infr.verifiers = classifiers
        infr.params['inference.enabled'] = enable_inference
        infr.params['autoreview.enabled'] = enable_autoreview

        infr.params['redun.pos'] = k_redun
        infr.params['redun.neg'] = k_redun

        # keeps track of edges where the decision != the groundtruth
        infr.mistake_edges = set()

        infr.queue = util.PriorityQueue()

        infr.oracle = UserOracle(oracle_accuracy, rng=infr.name)

        if match_state_thresh is None:
            match_state_thresh = {
                POSTV: 1.0,
                NEGTV: 1.0,
                INCMP: 1.0,
            }

        pb_state_thresh = None
        if pb_state_thresh is None:
            pb_state_thresh = {
                'pb': .5,
                'notpb': .9,
            }

        infr.task_thresh = {
            'photobomb_state': pd.Series(pb_state_thresh),
            'match_state': pd.Series(match_state_thresh)
        }
        infr.params['algo.max_outer_loops'] = max_outer_loops

    def init_test_mode(infr):
        from graphid.core import nx_dynamic_graph
        infr.print('init_test_mode')
        infr.test_mode = True
        # infr.edge_truth = {}
        infr.metrics_list = []
        infr.test_state = {
            'n_decision': 0,
            'n_algo': 0,
            'n_manual': 0,
            'n_true_merges': 0,
            'n_error_edges': 0,
            'confusion': None,
        }
        infr.test_gt_pos_graph = nx_dynamic_graph.DynConnGraph()
        infr.test_gt_pos_graph.add_nodes_from(infr.aids)

        infr.node_truth = infr.get_node_attrs('orig_name_label')
        infr.nid_to_gt_cc = ub.group_items(infr.node_truth.keys(),
                                           infr.node_truth.values())

        # infr.nid_to_gt_cc = ub.group_items(infr.aids, infr.orig_name_labels)
        # infr.node_truth = ub.dzip(infr.aids, infr.orig_name_labels)

        # infr.real_n_pcc_mst_edges = sum(
        #     len(cc) - 1 for cc in infr.nid_to_gt_cc.values())
        # util.cprint('real_n_pcc_mst_edges = %r' % (
        #     infr.real_n_pcc_mst_edges,), 'red')

        infr.metrics_list = []
        infr.real_n_pcc_mst_edges = sum(
            len(cc) - 1 for cc in infr.nid_to_gt_cc.values())
        infr.print('real_n_pcc_mst_edges = %r' % (
            infr.real_n_pcc_mst_edges,), color='red')

    def measure_error_edges(infr):
        for edge, data in infr.edges(data=True):
            true_state = data['truth']
            pred_state = data.get('evidence_decision', UNREV)
            if pred_state != UNREV:
                if true_state != pred_state:
                    error = ub.odict([('real', true_state),
                                      ('pred', pred_state)])
                    yield edge, error

    def measure_metrics(infr):
        real_pos_edges = []

        n_true_merges = infr.test_state['n_true_merges']
        confusion = infr.test_state['confusion']

        n_tp = confusion[POSTV][POSTV]
        confusion[POSTV]
        columns = set(confusion.keys())
        reviewd_cols = columns - {UNREV}
        non_postv = reviewd_cols - {POSTV}
        non_negtv = reviewd_cols - {NEGTV}

        n_fn = sum(ub.take(confusion[POSTV], non_postv))
        n_fp = sum(ub.take(confusion[NEGTV], non_negtv))

        n_error_edges = sum(confusion[r][c] + confusion[c][r] for r, c in
                            it.combinations(reviewd_cols, 2))
        # assert n_fn + n_fp == n_error_edges

        pred_n_pcc_mst_edges = n_true_merges

        # Find all annotations involved in a mistake
        assert n_error_edges == len(infr.mistake_edges)
        direct_mistake_aids = {a for edge in infr.mistake_edges for a in edge}
        mistake_nids = set(infr.node_labels(*direct_mistake_aids))
        mistake_aids = set(ub.flatten([infr.pos_graph.component(nid)
                                       for nid in mistake_nids]))

        pos_acc = pred_n_pcc_mst_edges / infr.real_n_pcc_mst_edges
        metrics = {
            'n_decision': infr.test_state['n_decision'],
            'n_manual': infr.test_state['n_manual'],
            'n_algo': infr.test_state['n_algo'],
            'phase': infr.loop_phase,
            'pos_acc': pos_acc,
            'n_merge_total': infr.real_n_pcc_mst_edges,
            'n_merge_remain': infr.real_n_pcc_mst_edges - n_true_merges,
            'n_true_merges': n_true_merges,
            'recovering': infr.is_recovering(),
            # 'recovering2': infr.test_state['recovering'],
            'merge_remain': 1 - pos_acc,
            'n_mistake_aids': len(mistake_aids),
            'frac_mistake_aids': len(mistake_aids) / len(infr.aids),
            'n_mistake_nids': len(mistake_nids),
            'n_errors': n_error_edges,
            'n_fn': n_fn,
            'n_fp': n_fp,
            'refresh_support': len(infr.refresh.manual_decisions),
            'pprob_any': infr.refresh.prob_any_remain(),
            'mu': infr.refresh._ewma,
            'test_action': infr.test_state['test_action'],
            'action': infr.test_state.get('action', None),
            'user_id': infr.test_state['user_id'],
            'pred_decision': infr.test_state['pred_decision'],
            'true_decision': infr.test_state['true_decision'],
            'n_neg_redun': infr.neg_redun_metagraph.number_of_edges(),
            'n_neg_redun1': (infr.neg_metagraph.number_of_edges() -
                             infr.neg_metagraph.number_of_selfloops()),
        }

        return metrics

    def _print_previous_loop_statistics(infr, count):
        # Print stats about what happend in the this loop
        history = infr.metrics_list[-count:]
        recover_blocks = ub.group_items([
            (k, sum(1 for i in g))
            for k, g in it.groupby(util.take_column(history, 'recovering'))
        ]).get(True, [])
        infr.print((
            'Recovery mode entered {} times, '
            'made {} recovery decisions.').format(
                len(recover_blocks), sum(recover_blocks)), color='green')
        testaction_hist = ub.dict_hist(util.take_column(history, 'test_action'))
        infr.print(
            'Test Action Histogram: {}'.format(
                ub.repr2(testaction_hist, si=True)), color='yellow')
        if infr.params['inference.enabled']:
            action_hist = ub.dict_hist(
                util.emap(frozenset, util.take_column(history, 'action')))
            infr.print(
                'Inference Action Histogram: {}'.format(
                    ub.repr2(action_hist, si=True)), color='yellow')
        infr.print(
            'Decision Histogram: {}'.format(ub.repr2(ub.dict_hist(
                util.take_column(history, 'pred_decision')
            ), si=True)), color='yellow')
        infr.print(
            'User Histogram: {}'.format(ub.repr2(ub.dict_hist(
                util.take_column(history, 'user_id')
            ), si=True)), color='yellow')

    def _dynamic_test_callback(infr, edge, decision, prev_decision, user_id):
        was_gt_pos = infr.test_gt_pos_graph.has_edge(*edge)

        # prev_decision = infr.get_edge_attr(edge, 'decision', default=UNREV)
        # prev_decision = list(infr.edge_decision_from([edge]))[0]

        true_decision = infr.edge_truth[edge]

        was_within_pred = infr.pos_graph.are_nodes_connected(*edge)
        was_within_gt = infr.test_gt_pos_graph.are_nodes_connected(*edge)
        was_reviewed = prev_decision != UNREV
        is_within_gt = was_within_gt
        was_correct = prev_decision == true_decision

        is_correct = true_decision == decision
        # print('prev_decision = {!r}'.format(prev_decision))
        # print('decision = {!r}'.format(decision))
        # print('true_decision = {!r}'.format(true_decision))

        test_print = partial(infr.print, level=2)
        def test_print(x, **kw):
            infr.print('[ACTION] ' + x, level=2, **kw)
        # test_print = lambda *a, **kw: None  # NOQA

        if decision == POSTV:
            if is_correct:
                if not was_gt_pos:
                    infr.test_gt_pos_graph.add_edge(*edge)
        elif was_gt_pos:
            test_print("UNDID GOOD POSITIVE EDGE", color='darkred')
            infr.test_gt_pos_graph.remove_edge(*edge)
            is_within_gt = infr.test_gt_pos_graph.are_nodes_connected(*edge)

        split_gt = is_within_gt != was_within_gt
        if split_gt:
            test_print("SPLIT A GOOD MERGE", color='darkred')
            infr.test_state['n_true_merges'] -= 1

        confusion = infr.test_state['confusion']
        if confusion is None:
            # initialize dynamic confusion matrix
            states = (POSTV, NEGTV, INCMP, UNREV, UNKWN)
            confusion = {r: {c: 0 for c in states} for r in states}
            infr.test_state['confusion'] = confusion

        if was_reviewed:
            confusion[true_decision][prev_decision] -= 1
            confusion[true_decision][decision] += 1
        else:
            confusion[true_decision][decision] += 1

        test_action = None
        action_color = None

        if is_correct:
            # CORRECT DECISION
            if was_reviewed:
                if prev_decision == decision:
                    test_action = 'correct duplicate'
                    action_color = 'darkyellow'
                else:
                    infr.mistake_edges.remove(edge)
                    test_action = 'correction'
                    action_color = 'darkgreen'
                    if decision == POSTV:
                        if not was_within_gt:
                            test_action = 'correction redid merge'
                            action_color = 'darkgreen'
                            infr.test_state['n_true_merges'] += 1
            else:
                if decision == POSTV:
                    if not was_within_gt:
                        test_action = 'correct merge'
                        action_color = 'darkgreen'
                        infr.test_state['n_true_merges'] += 1
                    else:
                        test_action = 'correct redundant positive'
                        action_color = 'darkblue'
                else:
                    if decision == NEGTV:
                        test_action = 'correct negative'
                        action_color = 'teal'
                    else:
                        test_action = 'correct uninferrable'
                        action_color = 'teal'
        else:
            action_color = 'darkred'
            # INCORRECT DECISION
            infr.mistake_edges.add(edge)
            if was_reviewed:
                if prev_decision == decision:
                    test_action = 'incorrect duplicate'
                elif was_correct:
                    test_action = 'incorrect undid good edge'
            else:
                if decision == POSTV:
                    if was_within_pred:
                        test_action = 'incorrect redundant merge'
                    else:
                        test_action = 'incorrect new merge'
                else:
                    test_action = 'incorrect new mistake'

        infr.test_state['test_action'] = test_action
        infr.test_state['pred_decision'] = decision
        infr.test_state['true_decision'] = true_decision
        infr.test_state['user_id'] = user_id
        infr.test_state['recovering'] = (infr.recover_graph.has_node(edge[0]) or
                                         infr.recover_graph.has_node(edge[1]))

        infr.test_state['n_decision'] += 1
        if user_id.startswith('algo'):
            infr.test_state['n_algo'] += 1
        elif user_id.startswith('user') or user_id == 'oracle':
            infr.test_state['n_manual'] += 1
        else:
            raise AssertionError('unknown user_id=%r' % (user_id,))

        test_print(test_action, color=action_color)
        assert test_action is not None, 'what happened?'


class UserOracle(object):
    def __init__(oracle, accuracy, rng):
        if isinstance(rng, six.string_types):
            rng = sum(map(ord, rng))
        rng = util.ensure_rng(rng, api='python')

        if isinstance(accuracy, tuple):
            oracle.normal_accuracy = accuracy[0]
            oracle.recover_accuracy = accuracy[1]
        else:
            oracle.normal_accuracy = accuracy
            oracle.recover_accuracy = accuracy

        oracle.rng = rng
        oracle.states = {POSTV, NEGTV, INCMP}

    def review(oracle, edge, truth, infr, accuracy=None):
        feedback = {
            'user_id': 'user:oracle',
            'confidence': 'absolutely_sure',
            'evidence_decision': None,
            'meta_decision': NULL,
            'timestamp_s1': util.get_timestamp('int', isutc=True),
            'timestamp_c1': util.get_timestamp('int', isutc=True),
            'timestamp_c2': util.get_timestamp('int', isutc=True),
            'tags': [],
        }
        is_recovering = infr.is_recovering()

        if accuracy is None:
            if is_recovering:
                accuracy = oracle.recover_accuracy
            else:
                accuracy = oracle.normal_accuracy

        # The oracle can get anything where the hardness is less than its
        # accuracy

        hardness = oracle.rng.random()
        error = accuracy < hardness

        if error:
            error_options = list(oracle.states - {truth} - {INCMP})
            observed = oracle.rng.choice(list(error_options))
        else:
            observed = truth
        if accuracy < 1.0:
            feedback['confidence'] = 'pretty_sure'
        if accuracy < .5:
            feedback['confidence'] = 'guessing'
        feedback['evidence_decision'] = observed
        if error:
            infr.print(
                'ORACLE ERROR real={} pred={} acc={:.2f} hard={:.2f}'.format(
                    truth, observed, accuracy, hardness), 2, color='red')
        return feedback

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/graphid/graphid.core/mixin_simulation.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
