import pandas as pd
import numpy as np
import ubelt as ub
from itertools import zip_longest
from os.path import join, dirname
import warnings
import colorsys


def multi_plot(xdata=None, ydata=[], **kwargs):
    r"""
    plots multiple lines, bars, etc...

    This is the big function that implements almost all of the heavy lifting in
    this file.  Any function not using this should probably find a way to use
    it. It is pretty general and relatively clean.

    Args:
        xdata (ndarray): can also be a list of arrays
        ydata (list or dict of ndarrays): can also be a single array
        **kwargs:
            Misc:
                fnum, pnum, use_legend, legend_loc
            Labels:
                xlabel, ylabel, title, figtitle
                ticksize, titlesize, legendsize, labelsize
            Grid:
                gridlinewidth, gridlinestyle
            Ticks:
                num_xticks, num_yticks, tickwidth, ticklength, ticksize
            Data:
                xmin, xmax, ymin, ymax, spread_list
                # can append _list to any of these
                # these can be dictionaries if ydata was also a dict

                plot_kw_keys = ['label', 'color', 'marker', 'markersize',
                    'markeredgewidth', 'linewidth', 'linestyle']
                any plot_kw key can be a scalar (corresponding to all ydatas),
                a list if ydata was specified as a list, or a dict if ydata was
                specified as a dict.

                kind = ['bar', 'plot', ...]
            if kind='plot':
                spread
            if kind='bar':
                stacked, width

    References:
        matplotlib.org/examples/api/barchart_demo.html

    Example:
        >>> xdata = [1, 2, 3, 4, 5]
        >>> ydata_list = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, np.nan, 2, 1], [4, 3, np.nan, 1, 0]]
        >>> kwargs = {'label': ['spamΣ', 'eggs', 'jamµ', 'pram'],  'linestyle': '-'}
        >>> #fig = multi_plot(xdata, ydata_list, title='$\phi_1(\\vec{x})$', xlabel='\nfds', **kwargs)
        >>> fig = multi_plot(xdata, ydata_list, title='ΣΣΣµµµ', xlabel='\nfdsΣΣΣµµµ', **kwargs)
        >>> show_if_requested()

    Example:
        >>> fig1 = multi_plot([1, 2, 3], [4, 5, 6])
        >>> fig2 = multi_plot([1, 2, 3], [4, 5, 6], fnum=4)
        >>> show_if_requested()
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    ydata_list = ydata

    if isinstance(ydata_list, dict):
        # Special case where ydata is a dictionary
        if isinstance(xdata, str):
            # Special-er case where xdata is specified in ydata
            xkey = xdata
            ykeys = set(ydata_list.keys()) - {xkey}
            xdata = ydata_list[xkey]
        else:
            ykeys = list(ydata_list.keys())
        # Normalize input
        ydata_list = list(ub.take(ydata_list, ykeys))
        kwargs['label_list'] = kwargs.get('label_list', ykeys)
    else:
        ykeys = None

    def is_listlike(data):
        flag = isinstance(data, (list, np.ndarray, tuple, pd.Series))
        flag &= hasattr(data, '__getitem__') and hasattr(data, '__len__')
        return flag

    def is_list_of_scalars(data):
        if is_listlike(data):
            if len(data) > 0 and not is_listlike(data[0]):
                return True
        return False

    def is_list_of_lists(data):
        if is_listlike(data):
            if len(data) > 0 and is_listlike(data[0]):
                return True
        return False

    # allow ydata_list to be passed without a container
    if is_list_of_scalars(ydata_list):
        ydata_list = [np.array(ydata_list)]

    if xdata is None:
        xdata = list(range(len(ydata_list[0])))

    num_lines = len(ydata_list)

    # Transform xdata into xdata_list
    if is_list_of_lists(xdata):
        xdata_list = [np.array(xd, copy=True) for xd in xdata]
    else:
        xdata_list = [np.array(xdata, copy=True)] * num_lines

    fnum = ensure_fnum(kwargs.get('fnum', None))
    pnum = kwargs.get('pnum', None)
    kind = kwargs.get('kind', 'plot')
    transpose = kwargs.get('transpose', False)

    def parsekw_list(key, kwargs, num_lines=num_lines, ykeys=ykeys):
        """ copies relevant plot commands into plot_list_kw """
        if key in kwargs:
            val_list = kwargs[key]
        elif key + '_list' in kwargs:
            warnings.warn('*_list is depricated, just use kwarg {}'.format(key))
            val_list = kwargs[key + '_list']
        elif key + 's' in kwargs:
            # hack, multiple ways to do something
            warnings.warn('*s depricated, just use kwarg {}'.format(key))
            val_list = kwargs[key + 's']
        else:
            val_list = None

        if val_list is not None:
            if isinstance(val_list, dict):
                if ykeys is None:
                    raise ValueError('ydata is not a dict, but a property was.')
                else:
                    val_list = [val_list[key] for key in ykeys]

            if not isinstance(val_list, list):
                val_list = [val_list] * num_lines

        return val_list

    # Parse out arguments to ax.plot
    plot_kw_keys = ['label', 'color', 'marker', 'markersize',
                    'markeredgewidth', 'linewidth', 'linestyle', 'alpha']
    # hackish / extra args that dont go to plot, but help
    extra_plot_kw_keys = ['spread_alpha', 'autolabel', 'edgecolor', 'fill']
    plot_kw_keys += extra_plot_kw_keys
    plot_ks_vals = [parsekw_list(key, kwargs) for key in plot_kw_keys]
    plot_list_kw = dict([
        (key, vals)
        for key, vals in zip(plot_kw_keys, plot_ks_vals) if vals is not None
    ])

    if 'color' not in plot_list_kw:
        plot_list_kw['color'] = distinct_colors(num_lines)

    if kind == 'plot':
        if 'marker' not in plot_list_kw:
            plot_list_kw['marker'] = distinct_markers(num_lines)
        if 'spread_alpha' not in plot_list_kw:
            plot_list_kw['spread_alpha'] = [.2] * num_lines

    if kind == 'bar':
        # Remove non-bar kwargs
        for key in ['markeredgewidth', 'linewidth', 'marker', 'markersize', 'linestyle']:
            plot_list_kw.pop(key, None)

        stacked = kwargs.get('stacked', False)
        width_key = 'height' if transpose else 'width'
        if 'width_list' in kwargs:
            plot_list_kw[width_key] = kwargs['width_list']
        else:
            width = kwargs.get('width', .9)
            # if width is None:
            #     # HACK: need variable width
            #     # width = np.mean(np.diff(xdata_list[0]))
            #     width = .9
            if not stacked:
                width /= num_lines
            #plot_list_kw['orientation'] = ['horizontal'] * num_lines
            plot_list_kw[width_key] = [width] * num_lines

    spread_list = kwargs.get('spread_list', None)
    if spread_list is None:
        pass

    # nest into a list of dicts for each line in the multiplot
    valid_keys = list(set(plot_list_kw.keys()) - set(extra_plot_kw_keys))
    valid_vals = list(ub.take(plot_list_kw, valid_keys))
    plot_kw_list = [dict(zip(valid_keys, vals)) for vals in zip(*valid_vals)]

    extra_kw_keys = [key for key in extra_plot_kw_keys if key in plot_list_kw]
    extra_kw_vals = list(ub.take(plot_list_kw, extra_kw_keys))
    extra_kw_list = [dict(zip(extra_kw_keys, vals)) for vals in zip(*extra_kw_vals)]

    # Get passed in axes or setup a new figure
    ax = kwargs.get('ax', None)
    if ax is None:
        doclf = kwargs.get('doclf', False)
        fig = figure(fnum=fnum, pnum=pnum, docla=False, doclf=doclf)
        ax = plt.gca()
    else:
        plt.sca(ax)
        fig = ax.figure

    # +---------------
    # Draw plot lines
    ydata_list = np.array(ydata_list)

    if transpose:
        if kind == 'bar':
            plot_func = ax.barh
        elif kind == 'plot':
            def plot_func(_x, _y, **kw):
                return ax.plot(_y, _x, **kw)
    else:
        plot_func = getattr(ax, kind)  # usually ax.plot

    assert len(ydata_list) > 0, 'no ydata'
    #assert len(extra_kw_list) == len(plot_kw_list), 'bad length'
    #assert len(extra_kw_list) == len(ydata_list), 'bad length'
    _iter = enumerate(zip_longest(xdata_list, ydata_list, plot_kw_list, extra_kw_list))
    for count, (_xdata, _ydata, plot_kw, extra_kw) in _iter:
        ymask = np.isfinite(_ydata)
        ydata_ = _ydata.compress(ymask)
        xdata_ = _xdata.compress(ymask)
        if kind == 'bar':
            if stacked:
                # Plot bars on top of each other
                xdata_ = xdata_
            else:
                # Plot bars side by side
                baseoffset = (width * num_lines) / 2
                lineoffset = (width * count)
                offset = baseoffset - lineoffset  # Fixeme for more histogram bars
                xdata_ = xdata_ - offset
            # width_key = 'height' if transpose else 'width'
            # plot_kw[width_key] = np.diff(xdata)
        objs = plot_func(xdata_, ydata_, **plot_kw)

        if kind == 'bar':
            if extra_kw is not None and 'edgecolor' in extra_kw:
                for rect in objs:
                    rect.set_edgecolor(extra_kw['edgecolor'])
            if extra_kw is not None and extra_kw.get('autolabel', False):
                # FIXME: probably a more cannonical way to include bar
                # autolabeling with tranpose support, but this is a hack that
                # works for now
                for rect in objs:
                    if transpose:
                        numlbl = width = rect.get_width()
                        xpos = width + ((_xdata.max() - _xdata.min()) * .005)
                        ypos = rect.get_y() + rect.get_height() / 2.
                        ha, va = 'left', 'center'
                    else:
                        numlbl = height = rect.get_height()
                        xpos = rect.get_x() + rect.get_width() / 2.
                        ypos = 1.05 * height
                        ha, va = 'center', 'bottom'
                    barlbl = '%.3f' % (numlbl,)
                    ax.text(xpos, ypos, barlbl, ha=ha, va=va)

        # print('extra_kw = %r' % (extra_kw,))
        if kind == 'plot' and extra_kw.get('fill', False):
            ax.fill_between(_xdata, ydata_, alpha=plot_kw.get('alpha', 1.0),
                            color=plot_kw.get('color', None))  # , zorder=0)

        if spread_list is not None:
            # Plots a spread around plot lines usually indicating standard
            # deviation
            _xdata = np.array(_xdata)
            spread = spread_list[count]
            ydata_ave = np.array(ydata_)
            y_data_dev = np.array(spread)
            y_data_max = ydata_ave + y_data_dev
            y_data_min = ydata_ave - y_data_dev
            ax = plt.gca()
            spread_alpha = extra_kw['spread_alpha']
            ax.fill_between(_xdata, y_data_min, y_data_max, alpha=spread_alpha,
                            color=plot_kw.get('color', None))  # , zorder=0)
    # L________________

    #max_y = max(np.max(y_data), max_y)
    #min_y = np.min(y_data) if min_y is None else min(np.min(y_data), min_y)

    ydata = _ydata  # HACK
    xdata = _xdata  # HACK
    if transpose:
        #xdata_list = ydata_list
        ydata = xdata
        # Hack / Fix any transpose issues
        def transpose_key(key):
            if key.startswith('x'):
                return 'y' + key[1:]
            elif key.startswith('y'):
                return 'x' + key[1:]
            elif key.startswith('num_x'):
                # hackier, fixme to use regex or something
                return 'num_y' + key[5:]
            elif key.startswith('num_y'):
                # hackier, fixme to use regex or something
                return 'num_x' + key[5:]
            else:
                return key
        kwargs = {transpose_key(key): val for key, val in kwargs.items()}

    # Setup axes labeling
    title      = kwargs.get('title', None)
    xlabel     = kwargs.get('xlabel', '')
    ylabel     = kwargs.get('ylabel', '')
    def none_or_unicode(text):
        return None if text is None else ub.ensure_unicode(text)

    xlabel = none_or_unicode(xlabel)
    ylabel = none_or_unicode(ylabel)
    title = none_or_unicode(title)

    # Initial integration with mpl rcParams standards
    mplrc = mpl.rcParams.copy()
    mplrc.update({
        # 'legend.fontsize': custom_figure.LEGEND_SIZE,
        # 'axes.titlesize': custom_figure.TITLE_SIZE,
        # 'axes.labelsize': custom_figure.LABEL_SIZE,
        # 'legend.facecolor': 'w',
        # 'font.family': 'sans-serif',
        # 'xtick.labelsize': custom_figure.TICK_SIZE,
        # 'ytick.labelsize': custom_figure.TICK_SIZE,
    })
    mplrc.update(kwargs.get('rcParams', {}))

    titlesize  = kwargs.get('titlesize',  mplrc['axes.titlesize'])
    labelsize  = kwargs.get('labelsize',  mplrc['axes.labelsize'])
    legendsize = kwargs.get('legendsize', mplrc['legend.fontsize'])
    xticksize = kwargs.get('ticksize', mplrc['xtick.labelsize'])
    yticksize = kwargs.get('ticksize', mplrc['ytick.labelsize'])
    family = kwargs.get('fontfamily', mplrc['font.family'])

    tickformat = kwargs.get('tickformat', None)
    ytickformat = kwargs.get('ytickformat', tickformat)
    xtickformat = kwargs.get('xtickformat', tickformat)

    # 'DejaVu Sans','Verdana', 'Arial'
    weight = kwargs.get('fontweight', None)
    if weight is None:
        weight = 'normal'

    labelkw = {
        'fontproperties': mpl.font_manager.FontProperties(
            weight=weight,
            family=family, size=labelsize)
    }
    ax.set_xlabel(xlabel, **labelkw)
    ax.set_ylabel(ylabel, **labelkw)

    tick_fontprop = mpl.font_manager.FontProperties(family=family,
                                                    weight=weight)

    if tick_fontprop is not None:
        for ticklabel in ax.get_xticklabels():
            ticklabel.set_fontproperties(tick_fontprop)
        for ticklabel in ax.get_yticklabels():
            ticklabel.set_fontproperties(tick_fontprop)
    if xticksize is not None:
        for ticklabel in ax.get_xticklabels():
            ticklabel.set_fontsize(xticksize)
    if yticksize is not None:
        for ticklabel in ax.get_yticklabels():
            ticklabel.set_fontsize(yticksize)

    if xtickformat is not None:
        # mpl.ticker.StrMethodFormatter  # newstyle
        # mpl.ticker.FormatStrFormatter  # oldstyle
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(xtickformat))
    if ytickformat is not None:
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(ytickformat))

    xtick_kw = ytick_kw = {
        'width': kwargs.get('tickwidth', None),
        'length': kwargs.get('ticklength', None),
    }
    xtick_kw = {k: v for k, v in xtick_kw.items() if v is not None}
    ytick_kw = {k: v for k, v in ytick_kw.items() if v is not None}
    ax.xaxis.set_tick_params(**xtick_kw)
    ax.yaxis.set_tick_params(**ytick_kw)

    #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))

    # Setup axes limits
    if 'xlim' in kwargs:
        xlim = kwargs['xlim']
        if xlim is not None:
            if 'xmin' not in kwargs and 'xmax' not in kwargs:
                kwargs['xmin'] = xlim[0]
                kwargs['xmax'] = xlim[1]
            else:
                raise ValueError('use xmax, xmin instead of xlim')
    if 'ylim' in kwargs:
        ylim = kwargs['ylim']
        if ylim is not None:
            if 'ymin' not in kwargs and 'ymax' not in kwargs:
                kwargs['ymin'] = ylim[0]
                kwargs['ymax'] = ylim[1]
            else:
                raise ValueError('use ymax, ymin instead of ylim')

    xmin = kwargs.get('xmin', ax.get_xlim()[0])
    xmax = kwargs.get('xmax', ax.get_xlim()[1])
    ymin = kwargs.get('ymin', ax.get_ylim()[0])
    ymax = kwargs.get('ymax', ax.get_ylim()[1])

    text_type = str

    if text_type(xmax) == 'data':
        xmax = max([xd.max() for xd in xdata_list])
    if text_type(xmin) == 'data':
        xmin = min([xd.min() for xd in xdata_list])

    # Setup axes ticks
    num_xticks = kwargs.get('num_xticks', None)
    num_yticks = kwargs.get('num_yticks', None)

    if num_xticks is not None:
        # TODO check if xdata is integral
        if xdata.dtype.kind == 'i':
            xticks = np.linspace(np.ceil(xmin), np.floor(xmax),
                                 num_xticks).astype(np.int32)
        else:
            xticks = np.linspace((xmin), (xmax), num_xticks)
        ax.set_xticks(xticks)
    if num_yticks is not None:
        if ydata.dtype.kind == 'i':
            yticks = np.linspace(np.ceil(ymin), np.floor(ymax),
                                 num_yticks).astype(np.int32)
        else:
            yticks = np.linspace((ymin), (ymax), num_yticks)
        ax.set_yticks(yticks)

    force_xticks = kwargs.get('force_xticks', None)
    if force_xticks is not None:
        xticks = np.array(sorted(ax.get_xticks().tolist() + force_xticks))
        ax.set_xticks(xticks)

    yticklabels = kwargs.get('yticklabels', None)
    if yticklabels is not None:
        # Hack ONLY WORKS WHEN TRANSPOSE = True
        # Overrides num_yticks
        ax.set_yticks(ydata)
        ax.set_yticklabels(yticklabels)

    xticklabels = kwargs.get('xticklabels', None)
    if xticklabels is not None:
        # Overrides num_xticks
        ax.set_xticks(xdata)
        ax.set_xticklabels(xticklabels)

    xtick_rotation = kwargs.get('xtick_rotation', None)
    if xtick_rotation is not None:
        [lbl.set_rotation(xtick_rotation)
         for lbl in ax.get_xticklabels()]
    ytick_rotation = kwargs.get('ytick_rotation', None)
    if ytick_rotation is not None:
        [lbl.set_rotation(ytick_rotation)
         for lbl in ax.get_yticklabels()]

    # Axis padding
    xpad = kwargs.get('xpad', None)
    ypad = kwargs.get('ypad', None)
    xpad_factor = kwargs.get('xpad_factor', None)
    ypad_factor = kwargs.get('ypad_factor', None)
    if xpad is None and xpad_factor is not None:
        xpad = (xmax - xmin) * xpad_factor
    if ypad is None and ypad_factor is not None:
        ypad = (ymax - ymin) * ypad_factor
    xpad = 0 if xpad is None else xpad
    ypad = 0 if ypad is None else ypad
    ypad_high = kwargs.get('ypad_high', ypad)
    ypad_low  = kwargs.get('ypad_low', ypad)
    xpad_high = kwargs.get('xpad_high', xpad)
    xpad_low  = kwargs.get('xpad_low', xpad)
    xmin, xmax = (xmin - xpad_low), (xmax + xpad_high)
    ymin, ymax = (ymin - ypad_low), (ymax + ypad_high)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    xscale          = kwargs.get('xscale', None)
    yscale          = kwargs.get('yscale', None)
    if yscale is not None:
        ax.set_yscale(yscale)
    if xscale is not None:
        ax.set_xscale(xscale)

    gridlinestyle = kwargs.get('gridlinestyle', None)
    gridlinewidth = kwargs.get('gridlinewidth', None)
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    if gridlinestyle:
        for line in gridlines:
            line.set_linestyle(gridlinestyle)
    if gridlinewidth:
        for line in gridlines:
            line.set_linewidth(gridlinewidth)

    # Setup title
    if title is not None:
        titlekw = {
            'fontproperties': mpl.font_manager.FontProperties(
                family=family,
                weight=weight,
                size=titlesize)
        }
        ax.set_title(title, **titlekw)

    use_legend   = kwargs.get('use_legend', 'label' in valid_keys)
    legend_loc   = kwargs.get('legend_loc', 'best')
    legend_alpha = kwargs.get('legend_alpha', 1.0)
    if use_legend:
        legendkw = {
            'alpha': legend_alpha,
            'fontproperties': mpl.font_manager.FontProperties(
                family=family,
                weight=weight,
                size=legendsize)
        }
        legend(loc=legend_loc, ax=ax, **legendkw)

    figtitle = kwargs.get('figtitle', None)
    if figtitle is not None:
        set_figtitle(figtitle, fontfamily=family, fontweight=weight,
                     size=kwargs.get('figtitlesize'))

    use_darkbackground = kwargs.get('use_darkbackground', None)
    lightbg = kwargs.get('lightbg', None)
    if lightbg is None:
        lightbg = True
    if use_darkbackground is None:
        use_darkbackground = not lightbg
    if use_darkbackground:
        _dark_background(force=use_darkbackground is True)
    # TODO: return better info
    return fig


def figure(fnum=None, pnum=(1, 1, 1), title=None, figtitle=None, doclf=False,
           docla=False, projection=None, **kwargs):
    """
    http://matplotlib.org/users/gridspec.html

    Args:
        fnum (int): fignum = figure number
        pnum (int, str, or tuple(int, int, int)): plotnum = plot tuple
        title (str):  (default = None)
        figtitle (None): (default = None)
        docla (bool): (default = False)
        doclf (bool): (default = False)

    Returns:
        mpl.Figure: fig

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fnum = 1
        >>> fig = figure(fnum, (2, 2, 1))
        >>> plt.gca().text(0.5, 0.5, "ax1", va="center", ha="center")
        >>> fig = figure(fnum, (2, 2, 2))
        >>> plt.gca().text(0.5, 0.5, "ax2", va="center", ha="center")
        >>> show_if_requested()

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fnum = 1
        >>> fig = figure(fnum, (2, 2, 1))
        >>> plt.gca().text(0.5, 0.5, "ax1", va="center", ha="center")
        >>> fig = figure(fnum, (2, 2, 2))
        >>> plt.gca().text(0.5, 0.5, "ax2", va="center", ha="center")
        >>> fig = figure(fnum, (2, 4, (1, slice(1, None))))
        >>> plt.gca().text(0.5, 0.5, "ax3", va="center", ha="center")
        >>> show_if_requested()
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def ensure_fig(fnum=None):
        if fnum is None:
            try:
                fig = plt.gcf()
            except Exception as ex:
                fig = plt.figure()
        else:
            try:
                fig = plt.figure(fnum)
            except Exception as ex:
                fig = plt.gcf()
        return fig

    def _convert_pnum_int_to_tup(int_pnum):
        # Convert pnum to tuple format if in integer format
        nr = int_pnum // 100
        nc = int_pnum // 10 - (nr * 10)
        px = int_pnum - (nr * 100) - (nc * 10)
        pnum = (nr, nc, px)
        return pnum

    def _pnum_to_subspec(pnum):
        if isinstance(pnum, str):
            pnum = list(pnum)
        nrow, ncols, plotnum = pnum
        # if kwargs.get('use_gridspec', True):
        # Convert old pnums to gridspec
        gs = gridspec.GridSpec(nrow, ncols)
        if isinstance(plotnum, (tuple, slice, list)):
            subspec = gs[plotnum]
        else:
            subspec = gs[plotnum - 1]
        return (subspec,)

    def _setup_subfigure(pnum):
        if isinstance(pnum, int):
            pnum = _convert_pnum_int_to_tup(pnum)
        axes_list = fig.get_axes()
        if docla or len(axes_list) == 0:
            if pnum is not None:
                assert pnum[0] > 0, 'nRows must be > 0: pnum=%r' % (pnum,)
                assert pnum[1] > 0, 'nCols must be > 0: pnum=%r' % (pnum,)
                subspec = _pnum_to_subspec(pnum)
                ax = fig.add_subplot(*subspec, projection=projection)
                if len(axes_list) > 0:
                    ax.cla()
            else:
                ax = plt.gca()
        else:
            if pnum is not None:
                subspec = _pnum_to_subspec(pnum)
                ax = plt.subplot(*subspec)
            else:
                ax = plt.gca()

    fig = ensure_fig(fnum)
    if doclf:
        fig.clf()
    if pnum is not None:
        _setup_subfigure(pnum)
    # Set the title / figtitle
    if title is not None:
        ax = plt.gca()
        ax.set_title(title)
    if figtitle is not None:
        fig.suptitle(figtitle)
    return fig


def pandas_plot_matrix(df, rot=90, ax=None, grid=True, label=None,
                       zerodiag=False,
                       cmap='viridis', showvals=False, logscale=True):
    import matplotlib as mpl
    import copy
    from matplotlib import pyplot as plt
    if ax is None:
        fig = figure(fnum=1, pnum=(1, 1, 1))
        fig.clear()
        ax = plt.gca()
    ax = plt.gca()
    values = df.values
    if zerodiag:
        values = values.copy()
        values = values - np.diag(np.diag(values))

    # aximg = ax.imshow(values, interpolation='none', cmap='viridis')
    if logscale:
        from matplotlib.colors import LogNorm
        vmin = df[df > 0].min().min()
        norm = LogNorm(vmin=vmin, vmax=values.max())
    else:
        norm = None

    cmap = copy.copy(mpl.cm.get_cmap(cmap))  # copy the default cmap
    cmap.set_bad((0, 0, 0))

    aximg = ax.matshow(values, interpolation='none', cmap=cmap, norm=norm)
    # aximg = ax.imshow(values, interpolation='none', cmap='viridis', norm=norm)

    # ax.imshow(values, interpolation='none', cmap='viridis')
    ax.grid(False)
    cax = plt.colorbar(aximg, ax=ax)
    if label is not None:
        cax.set_label(label)

    ax.set_xticks(list(range(len(df.index))))
    ax.set_xticklabels([lbl[0:100] for lbl in df.index])
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(rot)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment('center')

    ax.set_yticks(list(range(len(df.columns))))
    ax.set_yticklabels([lbl[0:100] for lbl in df.columns])
    for lbl in ax.get_yticklabels():
        lbl.set_horizontalalignment('right')
    for lbl in ax.get_yticklabels():
        lbl.set_verticalalignment('center')

    # Grid lines around the pixels
    if grid:
        offset = -.5
        xlim = [-.5, len(df.columns)]
        ylim = [-.5, len(df.index)]
        segments = []
        for x in range(ylim[1]):
            xdata = [x + offset, x + offset]
            ydata = ylim
            segment = list(zip(xdata, ydata))
            segments.append(segment)
        for y in range(xlim[1]):
            xdata = xlim
            ydata = [y + offset, y + offset]
            segment = list(zip(xdata, ydata))
            segments.append(segment)
        bingrid = mpl.collections.LineCollection(segments, color='w', linewidths=1)
        ax.add_collection(bingrid)

    if showvals:
        x_basis = np.arange(len(df.columns))
        y_basis = np.arange(len(df.index))
        x, y = np.meshgrid(x_basis, y_basis)

        for c, r in zip(x.flatten(), y.flatten()):
            val = df.iloc[r, c]
            ax.text(c, r, val, va='center', ha='center', color='white')
    return ax


def axes_extent(axs, pad=0.0):
    """
    Get the full extent of a group of axes, including axes labels, tick labels,
    and titles.
    """
    import itertools as it
    import matplotlib as mpl
    def axes_parts(ax):
        yield ax
        for label in ax.get_xticklabels():
            if label.get_text():
                yield label
        for label in ax.get_yticklabels():
            if label.get_text():
                yield label
        xlabel = ax.get_xaxis().get_label()
        ylabel = ax.get_yaxis().get_label()
        for label in (xlabel, ylabel, ax.title):
            if label.get_text():
                yield label

    items = it.chain.from_iterable(axes_parts(ax) for ax in axs)
    extents = [item.get_window_extent() for item in items]
    #mpl.transforms.Affine2D().scale(1.1)
    extent = mpl.transforms.Bbox.union(extents)
    extent = extent.expanded(1.0 + pad, 1.0 + pad)
    return extent


def extract_axes_extents(fig, combine=False, pad=0.0):
    # Make sure we draw the axes first so we can
    # extract positions from the text objects
    import matplotlib as mpl
    fig.canvas.draw()

    # Group axes that belong together
    atomic_axes = []
    seen_ = set([])
    for ax in fig.axes:
        if ax not in seen_:
            atomic_axes.append([ax])
            seen_.add(ax)

    dpi_scale_trans_inv = fig.dpi_scale_trans.inverted()
    axes_bboxes_ = [axes_extent(axs, pad) for axs in atomic_axes]
    axes_extents_ = [extent.transformed(dpi_scale_trans_inv) for extent in axes_bboxes_]
    # axes_extents_ = axes_bboxes_
    if combine:
        # Grab include extents of figure text as well
        # FIXME: This might break on OSX
        # http://stackoverflow.com/questions/22667224/bbox-backend
        renderer = fig.canvas.get_renderer()
        for mpl_text in fig.texts:
            bbox = mpl_text.get_window_extent(renderer=renderer)
            extent_ = bbox.expanded(1.0 + pad, 1.0 + pad)
            extent = extent_.transformed(dpi_scale_trans_inv)
            # extent = extent_
            axes_extents_.append(extent)
        axes_extents = mpl.transforms.Bbox.union(axes_extents_)
    else:
        axes_extents = axes_extents_
    # if True:
    #     axes_extents.x0 = 0
    #     # axes_extents.y1 = 0
    return axes_extents


def adjust_subplots(left=None, right=None, bottom=None, top=None, wspace=None,
                    hspace=None, fig=None):
    """
    Kwargs:
        left (float): left side of the subplots of the figure
        right (float): right side of the subplots of the figure
        bottom (float): bottom of the subplots of the figure
        top (float): top of the subplots of the figure
        wspace (float): width reserved for blank space between subplots
        hspace (float): height reserved for blank space between subplots
    """
    from matplotlib import pyplot as plt
    kwargs = dict(left=left, right=right, bottom=bottom, top=top,
                  wspace=wspace, hspace=hspace)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if fig is None:
        fig = plt.gcf()
    subplotpars = fig.subplotpars
    adjust_dict = subplotpars.__dict__.copy()
    del adjust_dict['validate']
    adjust_dict.update(kwargs)
    fig.subplots_adjust(**adjust_dict)


def dict_intersection(dict1, dict2):
    r"""
    Key AND Value based dictionary intersection

    Args:
        dict1 (dict):
        dict2 (dict):

    Returns:
        dict: mergedict_

    Example:
        >>> dict1 = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        >>> dict2 = {'b': 2, 'c': 3, 'd': 5, 'e': 21, 'f': 42}
        >>> mergedict_ = dict_intersection(dict1, dict2)
        >>> print(ub.urepr(mergedict_, nl=0, sort=1))
        {'b': 2, 'c': 3}
    """
    isect_keys = set(dict1.keys()).intersection(set(dict2.keys()))
    # maintain order if possible
    if isinstance(dict1, ub.odict):
        isect_keys_ = [k for k in dict1.keys() if k in isect_keys]
        _dict_cls = ub.odict
    else:
        isect_keys_ = isect_keys
        _dict_cls = dict
    dict_isect = _dict_cls(
        (k, dict1[k]) for k in isect_keys_ if dict1[k] == dict2[k]
    )
    return dict_isect


def _dark_background(ax=None, doubleit=False, force=False):
    r"""
    Args:
        ax (None): (default = None)
        doubleit (bool): (default = False)

    CommandLine:
        python -m .draw_func2 --exec-_dark_background --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> fig = figure()
        >>> _dark_background()
        >>> show_if_requested()
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    def is_using_style(style):
        style_dict = mpl.style.library[style]
        return len(dict_intersection(style_dict, mpl.rcParams)) == len(style_dict)

    if force:
        from mpl_toolkits.mplot3d import Axes3D
        BLACK = np.array((  0,   0,   0, 255)) / 255.0
        # Should use mpl style dark background instead
        bgcolor = BLACK * .9
        if ax is None:
            ax = plt.gca()
        if isinstance(ax, Axes3D):
            ax.set_axis_bgcolor(bgcolor)
            ax.tick_params(colors='white')
            return
        xy, width, height = _get_axis_xy_width_height(ax)
        if doubleit:
            halfw = (doubleit) * (width / 2)
            halfh = (doubleit) * (height / 2)
            xy = (xy[0] - halfw, xy[1] - halfh)
            width *= (doubleit + 1)
            height *= (doubleit + 1)
        rect = mpl.patches.Rectangle(xy, width, height, lw=0, zorder=0)
        rect.set_clip_on(True)
        rect.set_fill(True)
        rect.set_color(bgcolor)
        rect.set_zorder(-99999999999)
        rect = ax.add_patch(rect)


def _get_axis_xy_width_height(ax=None, xaug=0, yaug=0, waug=0, haug=0):
    """ gets geometry of a subplot """
    from matplotlib import pyplot as plt
    if ax is None:
        ax = plt.gca()
    autoAxis = ax.axis()
    xy     = (autoAxis[0] + xaug, autoAxis[2] + yaug)
    width  = (autoAxis[1] - autoAxis[0]) + waug
    height = (autoAxis[3] - autoAxis[2]) + haug
    return xy, width, height


_LEGEND_LOCATION = {
    'upper right':  1,
    'upper left':   2,
    'lower left':   3,
    'lower right':  4,
    'right':        5,
    'center left':  6,
    'center right': 7,
    'lower center': 8,
    'upper center': 9,
    'center':      10,
}


def set_figtitle(figtitle, subtitle='', forcefignum=True, incanvas=True,
                 size=None, fontfamily=None, fontweight=None,
                 fig=None):
    r"""
    Args:
        figtitle (?):
        subtitle (str): (default = '')
        forcefignum (bool): (default = True)
        incanvas (bool): (default = True)
        fontfamily (None): (default = None)
        fontweight (None): (default = None)
        size (None): (default = None)
        fig (None): (default = None)

    CommandLine:
        python -m .custom_figure set_figtitle --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> fig = figure(fnum=1, doclf=True)
        >>> result = set_figtitle(figtitle='figtitle', fig=fig)
        >>> # xdoc: +REQUIRES(--show)
        >>> show_if_requested()
    """
    from matplotlib import pyplot as plt
    if figtitle is None:
        figtitle = ''
    if fig is None:
        fig = plt.gcf()
    figtitle = ub.ensure_unicode(figtitle)
    subtitle = ub.ensure_unicode(subtitle)
    if incanvas:
        if subtitle != '':
            subtitle = '\n' + subtitle
        prop = {
            'family': fontfamily,
            'weight': fontweight,
            'size': size,
        }
        prop = {k: v for k, v in prop.items() if v is not None}
        sup = fig.suptitle(figtitle + subtitle)

        if prop:
            fontproperties = sup.get_fontproperties().copy()
            for key, val in prop.items():
                getattr(fontproperties, 'set_' + key)(val)
            sup.set_fontproperties(fontproperties)
            # fontproperties = mpl.font_manager.FontProperties(**prop)
    else:
        fig.suptitle('')
    # Set title in the window
    window_figtitle = ('fig(%d) ' % fig.number) + figtitle
    window_figtitle = window_figtitle.replace('\n', ' ')
    fig.canvas.set_window_title(window_figtitle)


def legend(loc='best', fontproperties=None, size=None, fc='w', alpha=1,
           ax=None, handles=None):
    r"""
    Args:
        loc (str): (default = 'best')
        fontproperties (None): (default = None)
        size (None): (default = None)

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> loc = 'best'
        >>> xdata = np.linspace(-6, 6)
        >>> ydata = np.sin(xdata)
        >>> plt.plot(xdata, ydata, label='sin')
        >>> fontproperties = None
        >>> size = None
        >>> result = legend(loc, fontproperties, size)
        >>> print(result)
        >>> show_if_requested()
    """
    from matplotlib import pyplot as plt
    assert loc in _LEGEND_LOCATION or loc == 'best', (
        'invalid loc. try one of %r' % (_LEGEND_LOCATION,))
    if ax is None:
        ax = plt.gca()
    if fontproperties is None:
        prop = {}
        if size is not None:
            prop['size'] = size
        # prop['weight'] = 'normal'
        # prop['family'] = 'sans-serif'
    else:
        prop = fontproperties
    legendkw = dict(loc=loc)
    if prop:
        legendkw['prop'] = prop
    if handles is not None:
        legendkw['handles'] = handles
    legend = ax.legend(**legendkw)
    if legend:
        legend.get_frame().set_fc(fc)
        legend.get_frame().set_alpha(alpha)


def distinct_colors(N, brightness=.878, randomize=True, hue_range=(0.0, 1.0), cmap_seed=None):
    r"""
    Args:
        N (int):
        brightness (float):

    Returns:
        list: RGB_tuples

    CommandLine:
        python -m color_funcs --test-distinct_colors --N 2 --show --hue-range=0.05,.95
        python -m color_funcs --test-distinct_colors --N 3 --show --hue-range=0.05,.95
        python -m color_funcs --test-distinct_colors --N 4 --show --hue-range=0.05,.95
        python -m .color_funcs --test-distinct_colors --N 3 --show --no-randomize
        python -m .color_funcs --test-distinct_colors --N 4 --show --no-randomize
        python -m .color_funcs --test-distinct_colors --N 6 --show --no-randomize
        python -m .color_funcs --test-distinct_colors --N 20 --show

    References:
        http://blog.jianhuashao.com/2011/09/generate-n-distinct-colors.html
    """
    # TODO: Add sin wave modulation to the sat and value
    # HACK for white figures
    from matplotlib import pyplot as plt
    import colorsys
    remove_yellow = True

    use_jet = False
    if use_jet:
        cmap = plt.cm.jet
        RGB_tuples = list(map(tuple, cmap(np.linspace(0, 1, N))))
    elif cmap_seed is not None:
        # Randomized map based on a seed
        #cmap_ = 'Set1'
        #cmap_ = 'Dark2'
        choices = [
            #'Set1', 'Dark2',
            'jet',
            #'gist_rainbow',
            #'rainbow',
            #'gnuplot',
            #'Accent'
        ]
        cmap_hack = ub.argval('--cmap-hack', default=None)
        ncolor_hack = ub.argval('--ncolor-hack', default=None)
        if cmap_hack is not None:
            choices = [cmap_hack]
        if ncolor_hack is not None:
            N = int(ncolor_hack)
            N_ = N
        seed = sum(list(map(ord, ub.hash_data(cmap_seed))))
        rng = np.random.RandomState(seed + 48930)
        cmap_str = rng.choice(choices, 1)[0]
        #print('cmap_str = %r' % (cmap_str,))
        cmap = plt.cm.get_cmap(cmap_str)
        #.hashstr27(cmap_seed)
        #cmap_seed = 0
        #pass
        jitter = (rng.randn(N) / (rng.randn(100).max() / 2)).clip(-1, 1) * ((1 / (N ** 2)))
        range_ = np.linspace(0, 1, N, endpoint=False)
        #print('range_ = %r' % (range_,))
        range_ = range_ + jitter
        #print('range_ = %r' % (range_,))
        while not (np.all(range_ >= 0) and np.all(range_ <= 1)):
            range_[range_ < 0] = np.abs(range_[range_ < 0] )
            range_[range_ > 1] = 2 - range_[range_ > 1]
        #print('range_ = %r' % (range_,))
        shift = rng.rand()
        range_ = (range_ + shift) % 1
        #print('jitter = %r' % (jitter,))
        #print('shift = %r' % (shift,))
        #print('range_ = %r' % (range_,))
        if ncolor_hack is not None:
            range_ = range_[0:N_]
        RGB_tuples = list(map(tuple, cmap(range_)))
    else:
        sat = brightness
        val = brightness
        hmin, hmax = hue_range
        if remove_yellow:
            hue_skips = [(.13, .24)]
        else:
            hue_skips = []
        hue_skip_ranges = [_[1] - _[0] for _ in hue_skips]
        total_skip = sum(hue_skip_ranges)
        hmax_ = hmax - total_skip
        hue_list = np.linspace(hmin, hmax_, N, endpoint=False, dtype=float)
        # Remove colors (like hard to see yellows) in specified ranges
        for skip, range_ in zip(hue_skips, hue_skip_ranges):
            hue_list = [hue if hue <= skip[0] else hue + range_ for hue in hue_list]
        HSV_tuples = [(hue, sat, val) for hue in hue_list]
        RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    if randomize:
        deterministic_shuffle(RGB_tuples)
    return RGB_tuples


def distinct_markers(num, style='astrisk', total=None, offset=0):
    r"""
    Args:
        num (?):
    """
    num_sides = 3
    style_num = {
        'astrisk': 2,
        'star': 1,
        'polygon': 0,
        'circle': 3
    }[style]
    if total is None:
        total = num
    total_degrees = 360 / num_sides
    marker_list = [
        (num_sides, style_num,  total_degrees * (count + offset) / total)
        for count in range(num)
    ]
    return marker_list


def deterministic_shuffle(list_, rng=0):
    r"""
    Args:
        list_ (list):
        seed (int):

    Returns:
        list: list_

    Example:
        >>> list_ = [1, 2, 3, 4, 5, 6]
        >>> seed = 1
        >>> list_ = deterministic_shuffle(list_, seed)
        >>> result = str(list_)
        >>> print(result)
        [3, 2, 5, 1, 4, 6]
    """
    from graphid import util
    rng = util.ensure_rng(rng)
    rng.shuffle(list_)
    return list_


_BASE_FNUM = 9001


def next_fnum(new_base=None):
    global _BASE_FNUM
    if new_base is not None:
        _BASE_FNUM = new_base
    _BASE_FNUM += 1
    return _BASE_FNUM


def ensure_fnum(fnum):
    if fnum is None:
        return next_fnum()
    return fnum


def _save_requested(fpath_, save_parts):
    raise NotImplementedError('havent done this yet')
    # dpi = ub.argval('--dpi', type_=int, default=200)
    from os.path import expanduser
    from matplotlib import pyplot as plt
    dpi = 200
    fpath_ = expanduser(fpath_)
    print('Figure save was requested')
    arg_dict = {}
    # HACK
    arg_dict = {
        key: (val[0] if len(val) == 1 else '[' + ']['.join(val) + ']')
        if isinstance(val, list) else val
        for key, val in arg_dict.items()
    }
    fpath_ = fpath_.format(**arg_dict)
    fpath_ = fpath_.replace(' ', '').replace('\'', '').replace('"', '')
    dpath = ub.argval('--dpath', type_=str, default=None)
    if dpath is None:
        gotdpath = False
        dpath = '.'
    else:
        gotdpath = True

    fpath = join(dpath, fpath_)
    if not gotdpath:
        dpath = dirname(fpath_)
    print('dpath = %r' % (dpath,))

    fig = plt.gcf()
    fig.dpi = dpi

    fpath_strict = ub.truepath(fpath)
    CLIP_WHITE = ub.argflag('--clipwhite')
    from graphid import util

    if save_parts:
        # TODO: call save_parts instead, but we still need to do the
        # special grouping.

        # Group axes that belong together
        atomic_axes = []
        seen_ = set([])
        for ax in fig.axes:
            div = _get_plotdat(ax, _DF2_DIVIDER_KEY, None)
            if div is not None:
                df2_div_axes = _get_plotdat_dict(ax).get('df2_div_axes', [])
                seen_.add(ax)
                seen_.update(set(df2_div_axes))
                atomic_axes.append([ax] + df2_div_axes)
                # TODO: pad these a bit
            else:
                if ax not in seen_:
                    atomic_axes.append([ax])
                    seen_.add(ax)

        hack_axes_group_row = ub.argflag('--grouprows')
        if hack_axes_group_row:
            groupid_list = []
            for axs in atomic_axes:
                for ax in axs:
                    groupid = ax.colNum
                groupid_list.append(groupid)

            groups = ub.group_items(atomic_axes, groupid_list)
            new_groups = list(map(ub.flatten, groups.values()))
            atomic_axes = new_groups
            #[[(ax.rowNum, ax.colNum) for ax in axs] for axs in atomic_axes]
            # save all rows of each column

        subpath_list = save_parts(fig=fig, fpath=fpath_strict,
                                  grouped_axes=atomic_axes, dpi=dpi)
        absfpath_ = subpath_list[-1]

        if CLIP_WHITE:
            for subpath in subpath_list:
                # remove white borders
                util.clipwhite_ondisk(subpath, subpath)
    else:
        savekw = {}
        # savekw['transparent'] = fpath.endswith('.png') and not noalpha
        savekw['transparent'] = ub.argflag('--alpha')
        savekw['dpi'] = dpi
        savekw['edgecolor'] = 'none'
        savekw['bbox_inches'] = extract_axes_extents(fig, combine=True)  # replaces need for clipwhite
        absfpath_ = ub.truepath(fpath)
        fig.savefig(absfpath_, **savekw)

        if CLIP_WHITE:
            # remove white borders
            fpath_in = fpath_out = absfpath_
            util.clipwhite_ondisk(fpath_in, fpath_out)

    if ub.argflag(('--diskshow', '--ds')):
        # show what we wrote
        ub.startfile(absfpath_)


def show_if_requested(N=1):
    """
    Used at the end of tests. Handles command line arguments for saving figures

    Referencse:
        http://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib

    """
    import matplotlib.pyplot as plt
    save_parts = ub.argflag('--saveparts')

    fpath_ = ub.argval('--save', default=None)
    if fpath_ is None:
        fpath_ = ub.argval('--saveparts', default=None)
        save_parts = True

    if fpath_ is not None:
        _save_requested(fpath_, save_parts)
    # elif ub.argflag('--cmd'):
    #     pass
    if ub.argflag('--show'):
        plt.show()


def save_parts(fig, fpath, grouped_axes=None, dpi=None):
    """
    FIXME: this works in mpl 2.0.0, but not 2.0.2

    Args:
        fig (?):
        fpath (str):  file path string
        dpi (None): (default = None)

    Returns:
        list: subpaths

    CommandLine:
        python -m draw_func2 save_parts

    Ignore:
        >>> # DISABLE_DOCTEST
        >>> import matplotlib as mpl
        >>> import matplotlib.pyplot as plt
        >>> def testimg(fname):
        >>>     return plt.imread(mpl.cbook.get_sample_data(fname))
        >>> fnames = ['grace_hopper.png', 'ada.png'] * 4
        >>> fig = plt.figure(1)
        >>> for c, fname in enumerate(fnames, start=1):
        >>>     ax = fig.add_subplot(3, 4, c)
        >>>     ax.imshow(testimg(fname))
        >>>     ax.set_title(fname[0:3] + str(c))
        >>>     ax.set_xticks([])
        >>>     ax.set_yticks([])
        >>> ax = fig.add_subplot(3, 1, 3)
        >>> ax.plot(np.sin(np.linspace(0, np.pi * 2)))
        >>> ax.set_xlabel('xlabel')
        >>> ax.set_ylabel('ylabel')
        >>> ax.set_title('title')
        >>> fpath = 'test_save_parts.png'
        >>> adjust_subplots(fig=fig, wspace=.3, hspace=.3, top=.9)
        >>> subpaths = save_parts(fig, fpath, dpi=300)
        >>> fig.savefig(fpath)
        >>> ub.startfile(subpaths[0])
        >>> ub.startfile(fpath)
    """
    if dpi:
        # Need to set figure dpi before we draw
        fig.dpi = dpi
    # We need to draw the figure before calling get_window_extent
    # (or we can figure out how to set the renderer object)
    # if getattr(fig.canvas, 'renderer', None) is None:
    fig.canvas.draw()

    # Group axes that belong together
    if grouped_axes is None:
        grouped_axes = []
        for ax in fig.axes:
            grouped_axes.append([ax])

    subpaths = []
    _iter = enumerate(grouped_axes, start=0)
    _iter = ub.ProgIter(list(_iter), label='save subfig')
    for count, axs in _iter:
        subpath = ub.augpath(fpath, suffix=chr(count + 65))
        extent = axes_extent(axs).transformed(fig.dpi_scale_trans.inverted())
        savekw = {}
        savekw['transparent'] = ub.argflag('--alpha')
        if dpi is not None:
            savekw['dpi'] = dpi
        savekw['edgecolor'] = 'none'
        fig.savefig(subpath, bbox_inches=extent, **savekw)
        subpaths.append(subpath)
    return subpaths


_qtensured = False


def qtensure():
    global _qtensured
    try:
        __IPYTHON__
    except NameError:
        return False
    else:
        import sys
        import IPython
        ipython = IPython.get_ipython()
        if ipython is None:
            # we must have exited ipython at some point
            return
        # import matplotlib as mpl

        if 'PyQt4' in sys.modules:
            #IPython.get_ipython().magic('pylab qt4')
            # if not mpl.get_backend().startswith('Qt4'):
            if not _qtensured:
                ipython.magic('pylab qt4 --no-import-all')
                _qtensured = True
        else:
            # if gt.__PYQT__.GUITOOL_PYQT_VERSION == 5:
            # if not mpl.get_backend().startswith('Qt5'):
            if not _qtensured:
                ipython.magic('pylab qt5 --no-import-all')
                _qtensured = True


def imshow(img, fnum=None, title=None, figtitle=None, pnum=None,
           interpolation='nearest', cmap=None, heatmap=False,
           data_colorbar=False, xlabel=None, redraw_image=True,
           colorspace='bgr', ax=None, alpha=None, norm=None, **kwargs):
    r"""
    Args:
        img (ndarray): image data
        fnum (int): figure number
        colorspace (str): if the data is 3-4 channels, this indicates the colorspace
            1 channel data is assumed grayscale. 4 channels assumes alpha.
        title (str):
        figtitle (None):
        pnum (tuple): plot number
        interpolation (str): other interpolations = nearest, bicubic, bilinear
        cmap (None):
        heatmap (bool):
        data_colorbar (bool):
        darken (None):
        redraw_image (bool): used when calling imshow over and over. if false
                                doesnt do the image part.

    Returns:
        tuple: (fig, ax)

    Kwargs:
        docla, doclf, projection

    Returns:
        tuple: (fig, ax)
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    if ax is not None:
        fig = ax.figure
        nospecial = True
    else:
        fig = figure(fnum=fnum, pnum=pnum, title=title, figtitle=figtitle, **kwargs)
        ax = plt.gca()
        nospecial = False
        #ax.set_xticks([])
        #ax.set_yticks([])
        #return fig, ax

    if not redraw_image:
        return fig, ax

    if isinstance(img, str):
        # Allow for path to image to be specified
        from graphid import util
        img_fpath = img
        img = util.imread(img_fpath)

    plt_imshow_kwargs = {
        'interpolation': interpolation,
        #'cmap': plt.get_cmap('gray'),
    }
    if alpha is not None:
        plt_imshow_kwargs['alpha'] = alpha

    if norm is not None:
        if norm is True:
            norm = mpl.colors.Normalize()
        plt_imshow_kwargs['norm'] = norm
    else:
        if cmap is None and not heatmap and not nospecial:
            plt_imshow_kwargs['vmin'] = 0
            plt_imshow_kwargs['vmax'] = 255
    if heatmap:
        cmap = 'hot'

    # Handle tensor chw format in most cases
    if img.ndim == 3:
        if img.shape[0] == 3 or img.shape[0] == 1:
            if img.shape[2] > 4:
                # probably in chw format
                img = img.transpose(1, 2, 0)

    try:
        if len(img.shape) == 3 and (img.shape[2] == 3 or img.shape[2] == 4):
            # img is in a color format
            from graphid import util

            dst_space = 'rgb'
            if img.shape[2] == 4:
                colorspace += 'a'
                dst_space += 'a'

            imgRGB = util.convert_colorspace(img, dst_space=dst_space,
                                             src_space=colorspace)
            if imgRGB.dtype.kind == 'f':
                maxval = imgRGB.max()
                if maxval > 1.01 and maxval < 256:
                    imgRGB = np.array(imgRGB, dtype=np.uint8)
            ax.imshow(imgRGB, **plt_imshow_kwargs)

        elif len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            # img is in grayscale
            if len(img.shape) == 3:
                imgGRAY = img.reshape(img.shape[0:2])
            else:
                imgGRAY = img
            if cmap is None:
                cmap = plt.get_cmap('gray')
            if isinstance(cmap, str):
                cmap = plt.get_cmap(cmap)
            # for some reason gray floats aren't working right
            if imgGRAY.max() <= 1.01 and imgGRAY.min() >= -1E-9:
                imgGRAY = (imgGRAY * 255).astype(np.uint8)
            ax.imshow(imgGRAY, cmap=cmap, **plt_imshow_kwargs)
        else:
            raise AssertionError(
                'unknown image format. img.dtype=%r, img.shape=%r' %
                (img.dtype, img.shape))
    except TypeError as te:
        print('[df2] imshow ERROR %r' % (te,))
        raise
    except Exception as ex:
        print('!!!!!!!!!!!!!!WARNING!!!!!!!!!!!')
        print('[df2] type(img) = %r' % type(img))
        if not isinstance(img, np.ndarray):
            print('!!!!!!!!!!!!!!ERRROR!!!!!!!!!!!')
            pass
            #print('img = %r' % (img,))
        print('[df2] img.dtype = %r' % (img.dtype,))
        print('[df2] type(img) = %r' % (type(img),))
        print('[df2] img.shape = %r' % (img.shape,))
        print('[df2] imshow ERROR %r' % ex)
        raise
    #plt.set_cmap('gray')
    ax.set_xticks([])
    ax.set_yticks([])

    if data_colorbar is True:
        scores = np.unique(img.flatten())
        if cmap is None:
            cmap = 'hot'
        colors = scores_to_color(scores, cmap)
        colorbar(scores, colors)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if figtitle is not None:
        set_figtitle(figtitle)
    return fig, ax


def colorbar(scalars, colors, custom=False, lbl=None, ticklabels=None,
             float_format='%.2f', **kwargs):
    """
    adds a color bar next to the axes based on specific scalars

    Args:
        scalars (ndarray):
        colors (ndarray):
        custom (bool): use custom ticks

    Kwargs:
        See plt.colorbar

    Returns:
        cb : matplotlib colorbar object

    Ignore:
        >>> scalars = np.array([-1, -2, 1, 1, 2, 7, 10])
        >>> cmap_ = 'plasma'
        >>> logscale = False
        >>> custom = True
        >>> reverse_cmap = True
        >>> val2_customcolor  = {
        ...        -1: UNKNOWN_PURP,
        ...        -2: LIGHT_BLUE,
        ...    }
        >>> colors = scores_to_color(scalars, cmap_=cmap_, logscale=logscale, reverse_cmap=reverse_cmap, val2_customcolor=val2_customcolor)
        >>> colorbar(scalars, colors, custom=custom)
        >>> df2.present()
        >>> show_if_requested()

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> scalars = np.linspace(0, 1, 100)
        >>> cmap_ = 'plasma'
        >>> logscale = False
        >>> custom = False
        >>> reverse_cmap = False
        >>> colors = scores_to_color(scalars, cmap_=cmap_, logscale=logscale,
        >>>                          reverse_cmap=reverse_cmap)
        >>> colors = [lighten_rgb(c, .3) for c in colors]
        >>> colorbar(scalars, colors, custom=custom)
        >>> df2.present()
        >>> show_if_requested()
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    assert len(scalars) == len(colors), 'scalars and colors must be corresponding'
    if len(scalars) == 0:
        return None
    # Parameters
    ax = plt.gca()
    divider = _ensure_divider(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    xy, width, height = _get_axis_xy_width_height(ax)
    #orientation = ['vertical', 'horizontal'][0]
    TICK_FONTSIZE = 8
    #
    # Create scalar mappable with cmap
    if custom:
        # FIXME: clean this code up and change the name custom
        # to be meaningful. It is more like: display unique colors
        unique_scalars, unique_idx = np.unique(scalars, return_index=True)
        unique_colors = np.array(colors)[unique_idx]
        #max_, min_ = unique_scalars.max(), unique_scalars.min()
        #extent_ = max_ - min_
        #bounds = np.linspace(min_, max_ + 1, extent_ + 2)
        listed_cmap = mpl.colors.ListedColormap(unique_colors)
        #norm = mpl.colors.BoundaryNorm(bounds, listed_cmap.N)
        #sm = mpl.cm.ScalarMappable(cmap=listed_cmap, norm=norm)
        sm = mpl.cm.ScalarMappable(cmap=listed_cmap)
        sm.set_array(np.linspace(0, 1, len(unique_scalars) + 1))
    else:
        sorted_scalars = sorted(scalars)
        listed_cmap = scores_to_cmap(scalars, colors)
        sm = plt.cm.ScalarMappable(cmap=listed_cmap)
        sm.set_array(sorted_scalars)
    # Use mapable object to create the colorbar
    #COLORBAR_SHRINK = .42  # 1
    #COLORBAR_PAD = .01  # 1
    #COLORBAR_ASPECT = np.abs(20 * height / (width))  # 1

    cb = plt.colorbar(sm, cax=cax, **kwargs)

    ## Add the colorbar to the correct label
    #axis = cb.ax.yaxis  # if orientation == 'horizontal' else cb.ax.yaxis
    #position = 'bottom' if orientation == 'horizontal' else 'right'
    #axis.set_ticks_position(position)

    # This line alone removes data
    # axis.set_ticks([0, .5, 1])
    if custom:
        ticks = np.linspace(0, 1, len(unique_scalars) + 1)
        if len(ticks) < 2:
            ticks += .5
        else:
            # SO HACKY
            ticks += (ticks[1] - ticks[0]) / 2

        if isinstance(unique_scalars, np.ndarray) and unique_scalars.dtype.kind == 'f':
            ticklabels = [float_format % scalar for scalar in unique_scalars]
        else:
            ticklabels = unique_scalars
        cb.set_ticks(ticks)  # tick locations
        cb.set_ticklabels(ticklabels)  # tick labels
    elif ticklabels is not None:
        ticks_ = cb.ax.get_yticks()
        mx = ticks_.max()
        mn = ticks_.min()
        ticks = np.linspace(mn, mx, len(ticklabels))
        cb.set_ticks(ticks)  # tick locations
        cb.set_ticklabels(ticklabels)
        #cb.ax.get_yticks()
        #cb.set_ticks(ticks)  # tick locations
        #cb.set_ticklabels(ticklabels)  # tick labels
    # _set_plotdat(cb.ax, 'viztype', 'colorbar-%s' % (lbl,))
    # _set_plotdat(cb.ax, 'sm', sm)
    # FIXME: Figure out how to make a maximum number of ticks
    # and to enforce them to be inside the data bounds
    cb.ax.tick_params(labelsize=TICK_FONTSIZE)
    # Sets current axis
    plt.sca(ax)
    if lbl is not None:
        cb.set_label(lbl)
    return cb


_DF2_DIVIDER_KEY = '_df2_divider'


def _get_plotdat(ax, key, default=None):
    """ returns internal property from a matplotlib axis """
    _plotdat = _get_plotdat_dict(ax)
    val = _plotdat.get(key, default)
    return val


def _set_plotdat(ax, key, val):
    """ sets internal property to a matplotlib axis """
    _plotdat = _get_plotdat_dict(ax)
    _plotdat[key] = val


def _del_plotdat(ax, key):
    """ sets internal property to a matplotlib axis """
    _plotdat = _get_plotdat_dict(ax)
    if key in _plotdat:
        del _plotdat[key]


def _get_plotdat_dict(ax):
    """ sets internal property to a matplotlib axis """
    if '_plotdat' not in ax.__dict__:
        ax.__dict__['_plotdat'] = {}
    plotdat_dict = ax.__dict__['_plotdat']
    return plotdat_dict


def _ensure_divider(ax):
    """ Returns previously constructed divider or creates one """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = _get_plotdat(ax, _DF2_DIVIDER_KEY, None)
    if divider is None:
        divider = make_axes_locatable(ax)
        _set_plotdat(ax, _DF2_DIVIDER_KEY, divider)
        orig_append_axes = divider.append_axes
        def df2_append_axes(divider, position, size, pad=None, add_to_figure=True, **kwargs):
            """ override divider add axes to register the divided axes """
            div_axes = _get_plotdat(ax, 'df2_div_axes', [])
            new_ax = orig_append_axes(position, size, pad=pad, add_to_figure=add_to_figure, **kwargs)
            div_axes.append(new_ax)
            _set_plotdat(ax, 'df2_div_axes', div_axes)
            return new_ax
        new_method = df2_append_axes.__get__(divider, divider.__class__)
        setattr(divider, 'append_axes', new_method)
    return divider


def scores_to_cmap(scores, colors=None, cmap_='hot'):
    import matplotlib as mpl
    if colors is None:
        colors = scores_to_color(scores, cmap_=cmap_)
    scores = np.array(scores)
    colors = np.array(colors)
    sortx = scores.argsort()
    sorted_colors = colors[sortx]
    # Make a listed colormap and mappable object
    listed_cmap = mpl.colors.ListedColormap(sorted_colors)
    return listed_cmap


def scores_to_color(score_list, cmap_='hot', logscale=False, reverse_cmap=False,
                    custom=False, val2_customcolor=None, score_range=None,
                    cmap_range=(.1, .9)):
    """
    Other good colormaps are 'spectral', 'gist_rainbow', 'gist_ncar', 'Set1',
    'Set2', 'Accent'
    # TODO: plasma

    Args:
        score_list (list):
        cmap_ (str): defaults to hot
        logscale (bool):
        cmap_range (tuple): restricts to only a portion of the cmap to avoid extremes

    Returns:
        <class '_ast.ListComp'>

    Ignore:
        >>> score_list = np.array([-1, -2, 1, 1, 2, 10])
        >>> cmap_ = 'hot'
        >>> logscale = False
        >>> reverse_cmap = True
        >>> custom = True
        >>> val2_customcolor  = {
        ...        -1: UNKNOWN_PURP,
        ...        -2: LIGHT_BLUE,
        ...    }
    """
    import matplotlib.pyplot as plt
    assert len(score_list.shape) == 1, 'score must be 1d'
    if len(score_list) == 0:
        return []

    def apply_logscale(scores):
        scores = np.array(scores)
        above_zero = scores >= 0
        scores_ = scores.copy()
        scores_[above_zero] = scores_[above_zero] + 1
        scores_[~above_zero] = scores_[~above_zero] - 1
        scores_ = np.log2(scores_)
        return scores_

    if logscale:
        # Hack
        score_list = apply_logscale(score_list)
    cmap = plt.get_cmap(cmap_)
    #else:
    #    cmap = cmap_
    if reverse_cmap:
        cmap = reverse_colormap(cmap)
    #if custom:
    #    base_colormap = cmap
    #    data = score_list
    #    cmap = customize_colormap(score_list, base_colormap)
    if score_range is None:
        min_ = score_list.min()
        max_ = score_list.max()
    else:
        min_ = score_range[0]
        max_ = score_range[1]
        if logscale:
            min_, max_ = apply_logscale([min_, max_])
    if cmap_range is None:
        cmap_scale_min, cmap_scale_max = 0., 1.
    else:
        cmap_scale_min, cmap_scale_max = cmap_range
    extent_ = max_ - min_
    if extent_ == 0:
        colors = [cmap(.5) for fx in range(len(score_list))]
    else:
        if False and logscale:
            # hack
            def score2_01(score):
                return np.log2(
                    1 + cmap_scale_min + cmap_scale_max *
                    (float(score) - min_) / (extent_))
            score_list = np.array(score_list)
            #rank_multiplier = score_list.argsort() / len(score_list)
            #normscore = np.array(list(map(score2_01, score_list))) * rank_multiplier
            normscore = np.array(list(map(score2_01, score_list)))
            colors =  list(map(cmap, normscore))
        else:
            def score2_01(score):
                return cmap_scale_min + cmap_scale_max * (float(score) - min_) / (extent_)
        colors = [cmap(score2_01(score)) for score in score_list]
        if val2_customcolor is not None:
            colors = [
                np.array(val2_customcolor.get(score, color))
                for color, score in zip(colors, score_list)]
    return colors


def reverse_colormap(cmap):
    """
    References:
        http://nbviewer.ipython.org/github/kwinkunks/notebooks/blob/master/Matteo_colourmaps.ipynb
    """
    import matplotlib as mpl
    if isinstance(cmap,  mpl.colors.ListedColormap):
        return mpl.colors.ListedColormap(cmap.colors[::-1])
    else:
        reverse = []
        k = []
        for key, channel in cmap._segmentdata.items():
            data = []
            for t in channel:
                data.append((1 - t[0], t[1], t[2]))
            k.append(key)
            reverse.append(sorted(data))
        cmap_reversed = mpl.colors.LinearSegmentedColormap(
            cmap.name + '_reversed', dict(zip(k, reverse)))
        return cmap_reversed


class PlotNums(object):
    """
    Convinience class for dealing with plot numberings (pnums)

    Example:
        >>> pnum_ = PlotNums(nRows=2, nCols=2)
        >>> # Indexable
        >>> print(pnum_[0])
        (2, 2, 1)
        >>> # Iterable
        >>> print(ub.urepr(list(pnum_), nl=0, nobr=1))
        (2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)
        >>> # Callable (iterates through a default iterator)
        >>> print(pnum_())
        (2, 2, 1)
        >>> print(pnum_())
        (2, 2, 2)
    """

    def __init__(self, nRows=None, nCols=None, nSubplots=None, start=0):
        nRows, nCols = self._get_num_rc(nSubplots, nRows, nCols)
        self.nRows = nRows
        self.nCols = nCols
        base = 0
        self.offset = 0 if base == 1 else 1
        self.start = start
        self._iter = None

    def __getitem__(self, px):
        return (self.nRows, self.nCols, px + self.offset)

    def __call__(self):
        """
        replacement for make_pnum_nextgen

        Example:
            >>> import itertools as it
            >>> pnum_ = PlotNums(nSubplots=9)
            >>> pnum_list = [pnum_() for _ in range(len(pnum_))]
            >>> result = ('pnum_list = %s' % (ub.urepr(pnum_list),))
            >>> print(result)

        Example:
            >>> import itertools as it
            >>> for nRows, nCols, nSubplots in it.product([None, 3], [None, 3], [None, 9]):
            >>>     start = 0
            >>>     pnum_ = PlotNums(nRows, nCols, nSubplots, start)
            >>>     pnum_list = [pnum_() for _ in range(len(pnum_))]
            >>>     print((nRows, nCols, nSubplots))
            >>>     result = ('pnum_list = %s' % (ub.urepr(pnum_list),))
            >>>     print(result)
        """
        if self._iter is None:
            self._iter = iter(self)
        return next(self._iter)

    def __iter__(self):
        r"""
        Yields:
            tuple : pnum

        Example:
            >>> pnum_ = iter(PlotNums(nRows=3, nCols=2))
            >>> result = ub.urepr(list(pnum_), nl=1, nobr=1)
            >>> print(result)
            (3, 2, 1),
            (3, 2, 2),
            (3, 2, 3),
            (3, 2, 4),
            (3, 2, 5),
            (3, 2, 6),

        Example:
            >>> nRows = 3
            >>> nCols = 2
            >>> pnum_ = iter(PlotNums(nRows, nCols, start=3))
            >>> result = ub.urepr(list(pnum_), nl=1, nobr=1)
            >>> print(result)
            (3, 2, 4),
            (3, 2, 5),
            (3, 2, 6),
        """
        for px in range(self.start, len(self)):
            yield self[px]

    def __len__(self):
        total_plots = self.nRows * self.nCols
        return total_plots

    @classmethod
    def _get_num_rc(cls, nSubplots=None, nRows=None, nCols=None):
        r"""
        Gets a constrained row column plot grid

        Args:
            nSubplots (None): (default = None)
            nRows (None): (default = None)
            nCols (None): (default = None)

        Returns:
            tuple: (nRows, nCols)

        Example:
            >>> cases = [
            >>>     dict(nRows=None, nCols=None, nSubplots=None),
            >>>     dict(nRows=2, nCols=None, nSubplots=5),
            >>>     dict(nRows=None, nCols=2, nSubplots=5),
            >>>     dict(nRows=None, nCols=None, nSubplots=5),
            >>> ]
            >>> for kw in cases:
            >>>     print('----')
            >>>     size = PlotNums._get_num_rc(**kw)
            >>>     if kw['nSubplots'] is not None:
            >>>         assert size[0] * size[1] >= kw['nSubplots']
            >>>     print('**kw = %s' % (ub.urepr(kw),))
            >>>     print('size = %r' % (size,))
        """
        if nSubplots is None:
            if nRows is None:
                nRows = 1
            if nCols is None:
                nCols = 1
        else:
            if nRows is None and nCols is None:
                nRows, nCols = PlotNums._get_square_row_cols(nSubplots)
            elif nRows is not None:
                nCols = int(np.ceil(nSubplots / nRows))
            elif nCols is not None:
                nRows = int(np.ceil(nSubplots / nCols))
        return nRows, nCols

    def _get_square_row_cols(nSubplots, max_cols=None, fix=False, inclusive=True):
        r"""
        Args:
            nSubplots (int):
            max_cols (int):

        Returns:
            tuple: (int, int)

        Example:
            >>> nSubplots = 9
            >>> nSubplots_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            >>> max_cols = None
            >>> rc_list = [PlotNums._get_square_row_cols(nSubplots, fix=True) for nSubplots in nSubplots_list]
            >>> print(repr(np.array(rc_list).T))
            array([[1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                   [1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4]])
        """
        if nSubplots == 0:
            return 0, 0
        if inclusive:
            rounder = np.ceil
        else:
            rounder = np.floor
        if fix:
            # This function is very broken, but it might have dependencies
            # this is the correct version
            nCols = int(rounder(np.sqrt(nSubplots)))
            nRows = int(rounder(nSubplots / nCols))
            return nRows, nCols
        else:
            # This is the clamped num cols version
            if max_cols is None:
                max_cols = 5
                if nSubplots in [4]:
                    max_cols = 2
                if nSubplots in [5, 6, 7]:
                    max_cols = 3
                if nSubplots in [8]:
                    max_cols = 4
            nCols = int(min(nSubplots, max_cols))
            #nCols = int(min(rounder(np.sqrt(nrids)), 5))
            nRows = int(rounder(nSubplots / nCols))
        return nRows, nCols


def draw_border(ax, color, lw=2, offset=None, adjust=True):
    'draws rectangle border around a subplot'
    if adjust:
        xy, width, height = _get_axis_xy_width_height(ax, -.7, -.2, 1, .4)
    else:
        xy, width, height = _get_axis_xy_width_height(ax)
    if offset is not None:
        xoff, yoff = offset
        xy = [xoff, yoff]
        height = - height - yoff
        width = width - xoff
    import matplotlib as mpl
    rect = mpl.patches.Rectangle(xy, width, height, lw=lw)
    rect = ax.add_patch(rect)
    rect.set_clip_on(False)
    rect.set_fill(False)
    rect.set_edgecolor(color)
    return rect


def draw_boxes(boxes, box_format='xywh', color='blue', labels=None,
               textkw=None, ax=None):
    """
    Args:
        boxes (list): list of coordindates in xywh, tlbr, or cxywh format
        box_format (str): specify how boxes are formated
            xywh is the top left x and y pixel width and height
            cxywh is the center xy pixel width and height
            tlbr is the top left xy and the bottom right xy
        color (str): edge color of the boxes
        labels (list): if specified, plots a text annotation on each box

    Example:
        >>> qtensure()  # xdoc: +SKIP
        >>> bboxes = [[.1, .1, .6, .3], [.3, .5, .5, .6]]
        >>> col = draw_boxes(bboxes)
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    if ax is None:
        ax = plt.gca()

    if not len(boxes):
        return

    boxes = np.asarray(boxes)

    if box_format == 'xywh':
        xywh = boxes
    elif box_format == 'cxywh':
        cx, cy, w, h = boxes.T[0:4]
        x1 = cx - (w / 2)
        y1 = cy - (h / 2)
        xywh = np.vstack([x1, y1, w, h]).T
    elif box_format == 'tlbr':
        x1, y1 = boxes.T[0:2]
        w, h = boxes.T[2:4] - boxes.T[0:2]
        xywh = np.vstack([x1, y1, w, h]).T
    else:
        raise KeyError(box_format)

    edgecolor = Color(color).as01('rgba')
    facecolor = Color((0, 0, 0, 0)).as01('rgba')

    rectkw = dict(ec=edgecolor, fc=facecolor, lw=2, linestyle='solid')

    patches = [mpl.patches.Rectangle((x, y), w, h, **rectkw)
               for x, y, w, h in xywh]
    col = mpl.collections.PatchCollection(patches, match_original=True)
    ax.add_collection(col)

    if labels:
        texts = []
        default_textkw = {
            'horizontalalignment': 'left',
            'verticalalignment': 'top',
            'backgroundcolor': (0, 0, 0, .3),
            'color': 'white',
            'fontproperties': mpl.font_manager.FontProperties(
                size=6, family='monospace'),
        }
        tkw = default_textkw.copy()
        if textkw is not None:
            tkw.update(textkw)
        for (x1, y1, w, h), label in zip(xywh, labels):
            texts.append((x1, y1, label, tkw))

        for (x1, y1, catname, tkw) in texts:
            ax.text(x1, y1, catname, **tkw)
    return col


def draw_line_segments(pts1, pts2, ax=None, **kwargs):
    """
    draws `N` line segments between `N` pairs of points

    Args:
        pts1 (ndarray): Nx2
        pts2 (ndarray): Nx2
        ax (None): (default = None)
        **kwargs: lw, alpha, colors

    Example:
        >>> pts1 = np.array([(.1, .8), (.6, .8)])
        >>> pts2 = np.array([(.6, .7), (.4, .1)])
        >>> figure(fnum=None)
        >>> draw_line_segments(pts1, pts2)
        >>> # xdoc: +REQUIRES(--show)
        >>> import matplotlib.pyplot as plt
        >>> ax = plt.gca()
        >>> ax.set_xlim(0, 1)
        >>> ax.set_ylim(0, 1)
        >>> show_if_requested()
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    if ax is None:
        ax = plt.gca()
    assert len(pts1) == len(pts2), 'unaligned'
    segments = [(xy1, xy2) for xy1, xy2 in zip(pts1, pts2)]
    linewidth = kwargs.pop('lw', kwargs.pop('linewidth', 1.0))
    alpha = kwargs.pop('alpha', 1.0)
    if 'color' in kwargs:
        kwargs['colors'] = kwargs['color']
        # mpl.colors.ColorConverter().to_rgb(kwargs['color'])
    line_group = mpl.collections.LineCollection(segments, linewidths=linewidth,
                                                alpha=alpha, **kwargs)
    ax.add_collection(line_group)


def make_heatmask(probs, cmap='plasma', with_alpha=True):
    """
    Colorizes a single-channel intensity mask (with an alpha channel)
    """
    import matplotlib as mpl
    assert len(probs.shape) == 2
    cmap_ = mpl.cm.get_cmap(cmap)
    from graphid import util
    probs = util.ensure_float01(probs)
    heatmask = cmap_(probs)
    if with_alpha:
        heatmask[:, :, 0:3] = heatmask[:, :, 0:3][:, :, ::-1]
        heatmask[:, :, 3] = probs
    return heatmask


class Color(ub.NiceRepr):
    """
    move to colorutil?

    Example:
        >>> print(Color('g'))
        >>> print(Color('orangered'))
        >>> print(Color('#AAAAAA').as255())
        >>> print(Color([0, 255, 0]))
        >>> print(Color([1, 1, 1.]))
        >>> print(Color([1, 1, 1]))
        >>> print(Color(Color([1, 1, 1])).as255())
        >>> print(Color(Color([1., 0, 1, 0])).ashex())
        >>> print(Color([1, 1, 1], alpha=255))
        >>> print(Color([1, 1, 1], alpha=255, space='lab'))
    """
    def __init__(self, color, alpha=None, space=None):
        if isinstance(color, Color):
            assert alpha is None
            assert space is None
            space = color.space
            color = color.color01
        else:
            color = self._ensure_color01(color)
            if alpha is not None:
                alpha = self._ensure_color01([alpha])[0]

        if space is None:
            space = 'rgb'

        # always normalize the color down to 01
        color01 = list(color)

        if alpha is not None:
            if len(color01) not in [1, 3]:
                raise ValueError('alpha already in color')
            color01 = color01 + [alpha]

        # correct space if alpha is given
        if len(color01) in [2, 4]:
            if not space.endswith('a'):
                space += 'a'

        self.color01 = color01

        self.space = space

    def __nice__(self):
        colorpart = ', '.join(['{:.2f}'.format(c) for c in self.color01])
        return self.space + ': ' + colorpart

    def ashex(self, space=None):
        c255 = self.as255(space)
        return '#' + ''.join(['{:02x}'.format(c) for c in c255])

    def as255(self, space=None):
        color = (np.array(self.as01(space)) * 255).astype(np.uint8)
        return tuple(map(int, color))

    def as01(self, space=None):
        """
        self = mplutil.Color('red')
        mplutil.Color('green').as01('rgba')

        """
        color = tuple(self.color01)
        if space is not None:
            if space == self.space:
                pass
            elif space == 'rgba' and self.space == 'rgb':
                color = color + (1,)
            elif space == 'bgr' and self.space == 'rgb':
                color = color[::-1]
            elif space == 'rgb' and self.space == 'bgr':
                color = color[::-1]
            else:
                assert False
        return tuple(map(float, color))

    @classmethod
    def _is_base01(channels):
        """ check if a color is in base 01 """
        def _test_base01(channels):
            tests01 = {
                'is_float': all([isinstance(c, (float, np.float64)) for c in channels]),
                'is_01': all([c >= 0.0 and c <= 1.0 for c in channels]),
            }
            return tests01
        if isinstance(channels, str):
            return False
        return all(_test_base01(channels).values())

    @classmethod
    def _is_base255(Color, channels):
        """ there is a one corner case where all pixels are 1 or less """
        if (all(c > 0.0 and c <= 255.0 for c in channels) and any(c > 1.0 for c in channels)):
            # Definately in 255 space
            return True
        else:
            # might be in 01 or 255
            return all(isinstance(c, int) for c in channels)

    @classmethod
    def _hex_to_01(Color, hex_color):
        """
        hex_color = '#6A5AFFAF'
        """
        assert hex_color.startswith('#'), 'not a hex string %r' % (hex_color,)
        parts = hex_color[1:].strip()
        color255 = tuple(int(parts[i: i + 2], 16) for i in range(0, len(parts), 2))
        assert len(color255) in [3, 4], 'must be length 3 or 4'
        return Color._255_to_01(color255)

    def _ensure_color01(Color, color):
        """ Infer what type color is and normalize to 01 """
        if isinstance(color, str):
            color = Color._string_to_01(color)
        elif Color._is_base255(color):
            color = Color._255_to_01(color)
        return color

    @classmethod
    def _255_to_01(Color, color255):
        """ converts base 255 color to base 01 color """
        return [channel / 255.0 for channel in color255]

    @classmethod
    def _string_to_01(Color, color):
        """
        mplutil.Color._string_to_01('green')
        mplutil.Color._string_to_01('red')

        """
        from matplotlib import colors as mcolors
        if color in mcolors.BASE_COLORS:
            color01 = mcolors.BASE_COLORS[color]
        elif color in mcolors.CSS4_COLORS:
            color_hex = mcolors.CSS4_COLORS[color]
            color01 = Color._hex_to_01(color_hex)
        elif color.startswith('#'):
            color01 = Color._hex_to_01(color)
        else:
            raise ValueError('unknown color=%r' % (color,))
        return color01

    @classmethod
    def named_colors(cls):
        from matplotlib import colors as mcolors
        names = sorted(list(mcolors.BASE_COLORS.keys()) + list(mcolors.CSS4_COLORS.keys()))
        return names

    @classmethod
    def distinct(cls, num, space='bgr'):
        """
        Make multiple distinct colors
        """
        import matplotlib as mpl
        import matplotlib._cm  as _cm
        cm = mpl.colors.LinearSegmentedColormap.from_list(
            'gist_rainbow', _cm.datad['gist_rainbow'],
            mpl.rcParams['image.lut'])
        distinct_colors = [
            np.array(cm(i / num)).tolist()[0:3][::-1]
            for i in range(num)
        ]
        if space == 'bgr':
            return [Color(c, 'bgr').as01(space=space) for c in distinct_colors]
        else:
            return distinct_colors

    def adjust_hsv(self, hue_adjust=0.0, sat_adjust=0.0, val_adjust=0.0):
        """
        Performs adaptive changes to the HSV values of the color.

        Args:
            hue_adjust (float): addative
            sat_adjust (float):
            val_adjust (float):

        Returns:
            list: new_rgb

        CommandLine:
            python -m graphid.util.mplutil Color.adjust_hsv

        Example:
            >>> rgb_list = [Color(c).as01() for c in ['pink', 'yellow', 'green']]
            >>> hue_adjust = -0.1
            >>> sat_adjust = +0.5
            >>> val_adjust = -0.1
            >>> # execute function
            >>> new_rgb_list = [Color(rgb).adjust_hsv(hue_adjust, sat_adjust, val_adjust) for rgb in rgb_list]
            >>> print(ub.urepr(new_rgb_list, nl=1, sv=True))
            [
                <Color(rgb: 0.90, 0.23, 0.75)>,
                <Color(rgb: 0.90, 0.36, 0.00)>,
                <Color(rgb: 0.24, 0.40, 0.00)>,
            ]
            >>> # xdoc: +REQUIRES(--show)
            >>> color_list = rgb_list + new_rgb_list
            >>> testshow_colors(color_list)

        Ignore:
            print(np.array([-.1, 0.0, .1, .5, .9, 1.0, 1.1]))
            print(np.array([-.1, 0.0, .1, .5, .9, 1.0, 1.1]) % 1.0)
            print(divmod(np.array([-.1, 0.0, .1, .5, .9, 1.0, 1.1]), 1.0))
            print(1 + np.array([-.1, 0.0, .1, .5, .9, 1.0, 1.1]) % 1.0)
        """
        rgb = list(self.as01(space='rgb'))
        alpha = None
        if len(rgb) == 4:
            (R, G, B, alpha) = rgb
        else:
            (R, G, B) = rgb
        hsv = colorsys.rgb_to_hsv(R, G, B)
        (H, S, V) = hsv
        H_new = (H + hue_adjust)
        if H_new > 0 or H_new < 1:
            # is there a way to more ellegantly get this?
            H_new %= 1.0
        S_new = max(min(S + sat_adjust, 1.0), 0.0)
        V_new = max(min(V + val_adjust, 1.0), 0.0)
        #print('hsv=%r' % (hsv,))
        hsv_new = (H_new, S_new, V_new)
        #print('hsv_new=%r' % (hsv_new,))
        new_rgb = colorsys.hsv_to_rgb(*hsv_new)
        if alpha is not None:
            new_rgb = list(new_rgb) + [alpha]
        return Color(new_rgb, space='rgb').convert(space=self.space)

    def convert(self, space):
        """
        Converts to a new colorspace
        """
        return Color(self.as01(space=space), space=space)


def zoom_factory(ax=None, zoomable_list=[], base_scale=1.1):
    """
    References:
        https://gist.github.com/tacaswell/3144287
        http://stackoverflow.com/questions/11551049/matplotlib-plot-zooming-with-scroll-wheel
    """
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    def zoom_fun(event):
        #print('zooming')
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        if xdata is None or ydata is None:
            return
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            raise NotImplementedError('event.button=%r' % (event.button,))
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        for zoomable in zoomable_list:
            zoom = zoomable.get_zoom()
            new_zoom = zoom / (scale_factor ** (1.2))
            zoomable.set_zoom(new_zoom)
        # Get distance from the cursor to the edge of the figure frame
        x_left = xdata - cur_xlim[0]
        x_right = cur_xlim[1] - xdata
        y_top = ydata - cur_ylim[0]
        y_bottom = cur_ylim[1] - ydata
        ax.set_xlim([xdata - x_left * scale_factor, xdata + x_right * scale_factor])
        ax.set_ylim([ydata - y_top * scale_factor, ydata + y_bottom * scale_factor])

        # ----
        ax.figure.canvas.draw()  # force re-draw

    fig = ax.get_figure()  # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event', zoom_fun)

    #return the function
    return zoom_fun


def pan_factory(ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    self = PanEvents(ax)
    ax = self.ax
    fig = ax.get_figure()  # get the figure of interest
    self.cidBP = fig.canvas.mpl_connect('button_press_event', self.pan_on_press)
    self.cidBR = fig.canvas.mpl_connect('button_release_event', self.pan_on_release)
    self.cidBM = fig.canvas.mpl_connect('motion_notify_event', self.pan_on_motion)
    # attach the call back
    return self


class PanEvents(object):
    def __init__(self, ax=None):
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None
        self.xzoom = True
        self.yzoom = True
        self.cidBP = None
        self.cidBR = None
        self.cidBM = None
        self.cidKeyP = None
        self.cidKeyR = None
        self.cidScroll = None
        self.ax = ax

    def pan_on_press(self, event):
        if event.button != 1:
            return
        ax = self.ax
        if event.inaxes != ax:
            return
        self.cur_xlim = ax.get_xlim()
        self.cur_ylim = ax.get_ylim()
        self.press = self.x0, self.y0, event.xdata, event.ydata
        self.x0, self.y0, self.xpress, self.ypress = self.press

    def pan_on_release(self, event):
        if event.button != 1:
            return
        ax = self.ax
        self.press = None
        ax.figure.canvas.draw()

    def pan_on_motion(self, event):
        ax = self.ax
        if self.press is None:
            return
        if event.inaxes != ax:
            return
        dx = event.xdata - self.xpress
        dy = event.ydata - self.ypress
        self.cur_xlim -= dx
        self.cur_ylim -= dy
        ax.set_xlim(self.cur_xlim)
        ax.set_ylim(self.cur_ylim)

        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()


def relative_text(pos, text, ax=None, offset=None, **kwargs):
    """
    Places text on axes in a relative position

    Args:
        pos (tuple): relative xy position
        text (str): text
        ax (None): (default = None)
        offset (None): (default = None)
        **kwargs: horizontalalignment, verticalalignment, roffset, ha, va,
                  fontsize, fontproperties, fontproperties, clip_on

    CommandLine:
        python -m graphid.util.mplutil relative_text --show

    Example:
        >>> from graphid import util
        >>> import matplotlib as mpl
        >>> x = .5
        >>> y = .5
        >>> util.figure()
        >>> txt = 'Hello World'
        >>> family = 'monospace'
        >>> family = 'CMU Typewriter Text'
        >>> fontproperties = mpl.font_manager.FontProperties(family=family,
        >>>                                                  size=42)
        >>> relative_text((x, y), txt, halign='center',
        >>>               fontproperties=fontproperties)
        >>> util.show_if_requested()
    """
    from matplotlib import pyplot as plt
    if pos == 'lowerleft':
        pos = (.01, .99)
        kwargs['halign'] = 'left'
        kwargs['valign'] = 'bottom'
    elif pos == 'upperleft':
        pos = (.01, .01)
        kwargs['halign'] = 'left'
        kwargs['valign'] = 'top'
    x, y = pos
    if ax is None:
        ax = plt.gca()
    if 'halign' in kwargs:
        kwargs['horizontalalignment'] = kwargs.pop('halign')
    if 'valign' in kwargs:
        kwargs['verticalalignment'] = kwargs.pop('valign')
    if 'ha' in kwargs:
        kwargs['horizontalalignment'] = kwargs.pop('ha')
    if 'va' in kwargs:
        kwargs['verticalalignment'] = kwargs.pop('va')
    xy, width, height = get_axis_xy_width_height(ax)
    x_, y_ = ((xy[0]) + x * width, (xy[1] + height) - y * height)
    if offset is not None:
        xoff, yoff = offset
        x_ += xoff
        y_ += yoff
    ax.text(x_, y_, text, **kwargs)


def get_axis_xy_width_height(ax=None, xaug=0, yaug=0, waug=0, haug=0):
    """ gets geometry of a subplot """
    from matplotlib import pyplot as plt
    if ax is None:
        ax = plt.gca()
    autoAxis = ax.axis()
    xy     = (autoAxis[0] + xaug, autoAxis[2] + yaug)
    width  = (autoAxis[1] - autoAxis[0]) + waug
    height = (autoAxis[3] - autoAxis[2]) + haug
    return xy, width, height

if __name__ == '__main__':
    import xdoctest
    xdoctest.doctest_module(__file__)
