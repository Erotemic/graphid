import itertools as it
import ubelt as ub
from graphid import util
from os.path import join


def run_demo():
    """
    CommandLine:
        python -m graphid.demo.demo_script run_demo --viz
        python -m graphid.demo.demo_script run_demo

    Example:
        >>> run_demo()
    """
    from graphid import demo
    import matplotlib as mpl
    TMP_RC = {
        'axes.titlesize': 12,
        'axes.labelsize': int(ub.argval('--labelsize', default=8)),
        'font.family': 'sans-serif',
        'font.serif': 'CMU Serif',
        'font.sans-serif': 'CMU Sans Serif',
        'font.monospace': 'CMU Typewriter Text',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        # 'legend.alpha': .8,
        'legend.fontsize': 12,
        'legend.facecolor': 'w',
    }
    mpl.rcParams.update(TMP_RC)
    # ---- Synthetic data params
    params = {
        'redun.pos': 2,
        'redun.neg': 2,
    }
    # oracle_accuracy = .98
    # oracle_accuracy = .90
    # oracle_accuracy = (.8, 1.0)
    oracle_accuracy = (.85, 1.0)
    # oracle_accuracy = 1.0

    # --- draw params

    VISUALIZE = ub.argflag('--viz')
    # QUIT_OR_EMEBED = 'embed'
    QUIT_OR_EMEBED = 'quit'
    def asint(p):
        return p if p is None else int(p)
    TARGET_REVIEW = asint(ub.argval('--target', default=None))
    START = asint(ub.argval('--start', default=None))
    END = asint(ub.argval('--end', default=None))

    # ------------------

    # rng = np.random.RandomState(42)
    # infr = demo.demodata_infr(num_pccs=4, size=3, size_std=1, p_incon=0)
    # infr = demo.demodata_infr(num_pccs=6, size=7, size_std=1, p_incon=0)
    # infr = demo.demodata_infr(num_pccs=3, size=5, size_std=.2, p_incon=0)
    infr = demo.demodata_infr(pcc_sizes=[5, 2, 4])
    infr.verbose = 100
    infr.ensure_cliques()
    infr.ensure_full()
    # Dummy scoring

    infr.init_simulation(oracle_accuracy=oracle_accuracy, name='run_demo')
    # infr_gt = infr.copy()
    dpath = ub.ensuredir(ub.Path('~/Desktop/demo').expand())
    if 0:
        ub.delete(dpath)
    ub.ensuredir(dpath)

    fig_counter = it.count(0)

    def show_graph(infr, title, final=False, selected_edges=None):
        from matplotlib import pyplot as plt
        if not VISUALIZE:
            return
        # TODO: rich colored text?
        latest = '\n'.join(infr.latest_logs())
        showkw = dict(
            # fontsize=infr.graph.graph['fontsize'],
            # fontname=infr.graph.graph['fontname'],
            show_unreviewed_edges=True,
            show_inferred_same=False,
            show_inferred_diff=False,
            outof=(len(infr.aids)),
            # show_inferred_same=True,
            # show_inferred_diff=True,
            selected_edges=selected_edges,
            show_labels=True,
            simple_labels=True,
            # show_recent_review=not final,
            show_recent_review=False,
            # splines=infr.graph.graph['splines'],
            reposition=False,
            # with_colorbar=True
        )
        verbose = infr.verbose
        infr.verbose = 0
        infr_ = infr.copy()
        infr_ = infr
        infr_.verbose = verbose
        infr_.show(pickable=True, verbose=0, **showkw)
        infr.verbose = verbose
        # print('status ' + ub.urepr(infr_.status()))
        # infr.show(**showkw)
        ax = plt.gca()
        ax.set_title(title, fontsize=20)
        fig = plt.gcf()
        # fontsize = 22
        fontsize = 12
        if True:
            # postprocess xlabel
            lines = []
            for line in latest.split('\n'):
                if False and line.startswith('ORACLE ERROR'):
                    lines += ['ORACLE ERROR']
                else:
                    lines += [line]
            latest = '\n'.join(lines)
            if len(lines) > 10:
                fontsize = 16
            if len(lines) > 12:
                fontsize = 14
            if len(lines) > 14:
                fontsize = 12
            if len(lines) > 18:
                fontsize = 10

            if len(lines) > 23:
                fontsize = 8

        if True:
            util.mplutil.adjust_subplots(top=.95, left=0, right=1, bottom=.45,
                                         fig=fig)
            ax.set_xlabel('\n' + latest)
            xlabel = ax.get_xaxis().get_label()
            xlabel.set_horizontalalignment('left')
            # xlabel.set_x(.025)
            # xlabel.set_x(-.6)
            xlabel.set_x(-2.0)
            # xlabel.set_fontname('CMU Typewriter Text')
            xlabel.set_fontname('Inconsolata')
            xlabel.set_fontsize(fontsize)
        ax.set_aspect('equal')

        # ax.xaxis.label.set_color('red')
        fpath = join(dpath, 'demo_{:04d}.png'.format(next(fig_counter)))
        fig.savefig(fpath, dpi=300,
                    # transparent=True,
                    edgecolor='none')

        # pt.save_figure(dpath=dpath, dpi=300)
        infr.latest_logs()

    if VISUALIZE:
        infr.update_visual_attrs(groupby='name_label')
        infr.set_node_attrs('pin', 'true')
        node_dict = infr.graph.nodes
        print(ub.urepr(node_dict[1]))

    if VISUALIZE:
        infr.latest_logs()
        # Pin Nodes into the target groundtruth position
        show_graph(infr, 'target-gt')

    print(ub.urepr(infr.status()))
    infr.clear_feedback()
    infr.clear_name_labels()
    infr.clear_edges()
    print(ub.urepr(infr.status()))
    infr.latest_logs()

    if VISUALIZE:
        infr.update_visual_attrs()

    infr.prioritize('prob_match')
    if VISUALIZE or TARGET_REVIEW is None or TARGET_REVIEW == 0:
        show_graph(infr, 'initial state')

    def on_new_candidate_edges(infr, edges):
        # hack updateing visual attrs as a callback
        if VISUALIZE:
            infr.update_visual_attrs()

    infr.on_new_candidate_edges = on_new_candidate_edges

    infr.params.update(**params)
    infr.refresh_candidate_edges()

    VIZ_ALL = (VISUALIZE and TARGET_REVIEW is None and START is None)
    print('VIZ_ALL = %r' % (VIZ_ALL,))

    if VIZ_ALL or TARGET_REVIEW == 0:
        show_graph(infr, 'find-candidates')

    # _iter2 = enumerate(infr.generate_reviews(**params))
    # _iter2 = list(_iter2)
    # assert len(_iter2) > 0

    # prog = ub.ProgIter(_iter2, label='run_demo', bs=False, adjust=False,
    #                    enabled=False)
    count = 1
    first = 1
    for edge, priority in infr._generate_reviews(data=True):
        msg = 'review #%d, priority=%.3f' % (count, priority)
        print('\n----------')
        infr.print('pop edge {} with priority={:.3f}'.format(edge, priority))
        # print('remaining_reviews = %r' % (infr.remaining_reviews()),)
        # Make the next review

        if START is not None:
            VIZ_ALL = count >= START

        if END is not None and count >= END:
            break

        infr.print(msg)
        if ub.allsame(infr.pos_graph.node_labels(*edge)) and first:
            # Have oracle make a mistake early
            feedback = infr.request_oracle_review(edge, accuracy=0)
            first -= 1
        else:
            feedback = infr.request_oracle_review(edge)

        AT_TARGET = TARGET_REVIEW is not None and count >= TARGET_REVIEW - 1

        SHOW_CANDIATE_POP = True
        if SHOW_CANDIATE_POP and (VIZ_ALL or AT_TARGET):
            infr.print(ub.urepr(infr.task_probs['match_state'][edge], precision=4, si=True))
            infr.print('len(queue) = %r' % (len(infr.queue)))
            # Show edge selection
            infr.print('Oracle will predict: ' + feedback['evidence_decision'])
            show_graph(infr, 'pre' + msg, selected_edges=[edge])

        if count == TARGET_REVIEW:
            infr.EMBEDME = QUIT_OR_EMEBED == 'embed'
        infr.add_feedback(edge, **feedback)
        infr.print('len(queue) = %r' % (len(infr.queue)))
        # infr.apply_nondynamic_update()
        # Show the result
        if VIZ_ALL or AT_TARGET:
            show_graph(infr, msg)
            # import sys
            # sys.exit(1)
        if count == TARGET_REVIEW:
            break
        count += 1

    infr.print('status = ' + ub.urepr(infr.status(extended=False)))
    show_graph(infr, 'post-review (#reviews={})'.format(count), final=True)

    if VISUALIZE:
        if not getattr(infr, 'EMBEDME', False):
            # import plottool_ibeis as pt
            # util.mplutil.all_figures_tile()
            util.mplutil.show_if_requested()


if __name__ == '__main__':
    """
    CommandLine:
        python -m graphid.demo.demo_script all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
