"""
Port of the less useful parts of plottool that util_graphviz still depends on

TODO: try and depricate or refactor these
"""
import numpy as np


BLACK = np.array([0., 0., 0., 1.])
NEUTRAL_BLUE = np.array([0.62352941, 0.62352941, 0.94509804, 1.        ])


def get_plotdat_dict(ax):
    """ sets internal property to a matplotlib axis """
    if '_plotdat' not in ax.__dict__:
        ax.__dict__['_plotdat'] = {}
    plotdat_dict = ax.__dict__['_plotdat']
    return plotdat_dict


def set_plotdat(ax, key, val):
    """ sets internal property to a matplotlib axis """
    _plotdat = get_plotdat_dict(ax)
    _plotdat[key] = val


def make_bbox(bbox, theta=0, bbox_color=None, ax=None, lw=2, alpha=1.0,
              align='center', fill=None, **kwargs):
    if ax is None:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        ax = plt.gca()
    (rx, ry, rw, rh) = bbox
    # Transformations are specified in backwards order.
    trans_annotation = mpl.transforms.Affine2D()
    if align == 'center':
        trans_annotation.scale(rw, rh)
    elif align == 'outer':
        trans_annotation.scale(rw + (lw / 2), rh + (lw / 2))
    elif align == 'inner':
        trans_annotation.scale(rw - (lw / 2), rh - (lw / 2))

    trans_annotation.rotate(theta)
    trans_annotation.translate(rx + rw / 2, ry + rh / 2)
    t_end = trans_annotation + ax.transData
    bbox = mpl.patches.Rectangle((-.5, -.5), 1, 1, lw=lw, transform=t_end, **kwargs)
    bbox.set_fill(fill if fill else None)
    bbox.set_alpha(alpha)
    #bbox.set_transform(trans)
    bbox.set_edgecolor(bbox_color)
    return bbox


def get_axis_xy_width_height(ax=None, xaug=0, yaug=0, waug=0, haug=0):
    ' gets geometry of a subplot '
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    autoAxis = ax.axis()
    xy = ((autoAxis[0] + xaug), (autoAxis[2] + yaug))
    width = ((autoAxis[1] - autoAxis[0]) + waug)
    height = ((autoAxis[3] - autoAxis[2]) + haug)
    return (xy, width, height)


def ax_absolute_text(x_, y_, txt, ax=None, roffset=None, **kwargs):
    """
    Base function for text

    Kwargs:
        horizontalalignment in ['right', 'center', 'left'],
        verticalalignment in ['top']
        color
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    gca = plt.gca()
    kwargs = kwargs.copy()
    if (ax is None):
        ax = gca()
    if ('ha' in kwargs):
        kwargs['horizontalalignment'] = kwargs['ha']
    if ('va' in kwargs):
        kwargs['verticalalignment'] = kwargs['va']
    if ('fontproperties' not in kwargs):
        if ('fontsize' in kwargs):
            fontsize = kwargs['fontsize']
            font_prop = mpl.font_manager.FontProperties(family='monospace', size=fontsize)
            kwargs['fontproperties'] = font_prop
        else:
            kwargs['fontproperties'] = mpl.font_manager.FontProperties(family='monospace')
    if ('clip_on' not in kwargs):
        kwargs['clip_on'] = True
    if (roffset is not None):
        (xroff, yroff) = roffset
        (xy, width, height) = get_axis_xy_width_height(ax)
        x_ += (xroff * width)
        y_ += (yroff * height)
    return ax.text(x_, y_, txt, **kwargs)


def cartoon_stacked_rects(xy, width, height, num=4, shift=None, **kwargs):
    """
    pt.figure()
    xy = (.5, .5)
    width = .2
    height = .2
    ax = pt.gca()
    ax.add_collection(col)
    """
    import matplotlib as mpl
    if shift is None:
        shift = np.array([-width, height]) * (.1 / num)
    xy = np.array(xy)
    rectkw = dict(
        ec=kwargs.pop('ec', None),
        lw=kwargs.pop('lw', None),
        linestyle=kwargs.pop('linestyle', None),
    )
    patch_list = [mpl.patches.Rectangle(xy + shift * count, width, height,
                                        **rectkw)
                  for count in reversed(range(num))]
    col = mpl.collections.PatchCollection(patch_list, **kwargs)
    return col


def parse_fontkw(**kwargs):
    r"""
    Kwargs:
        fontsize, fontfamilty, fontproperties

    Example:
        >>> # xdoctest: +REQUIRES(module:matplotlib)
        >>> parse_fontkw()
    """
    from matplotlib.font_manager import FontProperties
    import matplotlib as mpl
    if 'fontproperties' not in kwargs:
        size = kwargs.get('fontsize', 14)
        weight = kwargs.get('fontweight', 'normal')
        fontname = kwargs.get('fontname', None)
        if fontname is not None:
            # TODO catch user warning
            '/usr/share/fonts/truetype/'
            '/usr/share/fonts/opentype/'
            fontpath = mpl.font_manager.findfont(fontname,
                                                 fallback_to_default=False)
            font_prop = FontProperties(fname=fontpath, weight=weight, size=size)
        else:
            family = kwargs.get('fontfamilty', 'monospace')
            font_prop = FontProperties(family=family, weight=weight, size=size)
    else:
        font_prop = kwargs['fontproperties']
    return font_prop
