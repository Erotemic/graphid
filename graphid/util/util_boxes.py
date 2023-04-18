import numpy as np
import ubelt as ub


def box_ious_py(boxes1, boxes2, bias=1):
    """
    This is the fastest python implementation of bbox_ious I found
    """
    w1 = boxes1[:, 2] - boxes1[:, 0] + bias
    h1 = boxes1[:, 3] - boxes1[:, 1] + bias
    w2 = boxes2[:, 2] - boxes2[:, 0] + bias
    h2 = boxes2[:, 3] - boxes2[:, 1] + bias

    areas1 = w1 * h1
    areas2 = w2 * h2

    x_maxs = np.minimum(boxes1[:, 2][:, None], boxes2[:, 2])
    x_mins = np.maximum(boxes1[:, 0][:, None], boxes2[:, 0])

    iws = np.maximum(x_maxs - x_mins + bias, 0)
    # note: it would be possible to significantly reduce the computation by
    # filtering any box pairs where iws <= 0. Not sure how to do with numpy.

    y_maxs = np.minimum(boxes1[:, 3][:, None], boxes2[:, 3])
    y_mins = np.maximum(boxes1[:, 1][:, None], boxes2[:, 1])

    ihs = np.maximum(y_maxs - y_mins + bias, 0)

    areas_sum = (areas1[:, None] + areas2)

    inter_areas = iws * ihs
    union_areas = (areas_sum - inter_areas)
    ious = inter_areas / union_areas
    return ious


class Boxes(ub.NiceRepr):
    """
    Converts boxes between different formats as long as the last dimension
    contains 4 coordinates and the format is specified.

    This is a convinience class, and should not not store the data for very
    long. The general idiom should be create class, convert data, and then get
    the raw data and let the class be garbage collected. This will help ensure
    that your code is portable and understandable if this class is not
    available.

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> Boxes([25, 30, 15, 10], 'xywh')
        <Boxes(xywh, array([25, 30, 15, 10]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').to_xywh()
        <Boxes(xywh, array([25, 30, 15, 10]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').to_cxywh()
        <Boxes(cxywh, array([32.5, 35. , 15. , 10. ]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').to_tlbr()
        <Boxes(tlbr, array([25, 30, 40, 40]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').scale(2).to_tlbr()
        <Boxes(tlbr, array([50., 60., 80., 80.]))>

    Example:
        >>> datas = [
        >>>     [1, 2, 3, 4],
        >>>     [[1, 2, 3, 4], [4, 5, 6, 7]],
        >>>     [[[1, 2, 3, 4], [4, 5, 6, 7]]],
        >>> ]
        >>> formats = ['xywh', 'cxywh', 'tlbr']
        >>> for format1 in formats:
        >>>     for data in datas:
        >>>         self = box1 = Boxes(data, format1)
        >>>         for format2 in formats:
        >>>             box2 = box1.toformat(format2)
        >>>             back = box2.toformat(format1)
        >>>             assert box1 == back
    """
    def __init__(self, data, format='xywh'):
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        self.data = data
        self.format = format

    def __eq__(self, other):
        return np.all(self.data == other.data) and self.format == other.format

    def __nice__(self):
        # return self.format + ', shape=' + str(list(self.data.shape))
        data_repr = repr(self.data)
        if '\n' in data_repr:
            data_repr = ub.indent('\n' + data_repr.lstrip('\n'), '    ')
        return '{}, {}'.format(self.format, data_repr)

    __repr__ = ub.NiceRepr.__str__

    @classmethod
    def random(Boxes, num=1, scale=1.0, format='xywh', rng=None):
        """
        Makes random boxes

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> Boxes.random(3, rng=0, scale=100)
            <Boxes(xywh,
                array([[27, 35, 30, 27],
                       [21, 32, 21, 44],
                       [48, 19, 39, 26]]))>
        """
        from graphid import util
        rng = util.ensure_rng(rng)

        xywh = (rng.rand(num, 4) * scale / 2)
        as_integer = isinstance(scale, int)
        if as_integer:
            xywh = xywh.astype(int)
        boxes = Boxes(xywh, format='xywh').toformat(format, copy=False)
        return boxes

    def copy(self):
        new_data = self.data.copy()
        return Boxes(new_data, self.format)

    def scale(self, factor):
        r"""
        works with tlbr, cxywh, xywh, xy, or wh formats

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> Boxes(np.array([1, 1, 10, 10])).scale(2).data
            array([ 2.,  2., 20., 20.])
            >>> Boxes(np.array([[1, 1, 10, 10]])).scale((2, .5)).data
            array([[ 2. ,  0.5, 20. ,  5. ]])
            >>> Boxes(np.array([[10, 10]])).scale(.5).data
            array([[5., 5.]])
        """
        boxes = self.data
        sx, sy = factor if ub.iterable(factor) else (factor, factor)
        if boxes.dtype.kind != 'f':
            new_data = boxes.astype(float)
        else:
            new_data = boxes.copy()
        new_data[..., 0:4:2] *= sx
        new_data[..., 1:4:2] *= sy
        return Boxes(new_data, self.format)

    def shift(self, amount):
        """
        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> Boxes([25, 30, 15, 10], 'xywh').shift(10)
            <Boxes(xywh, array([35., 40., 15., 10.]))>
            >>> Boxes([25, 30, 15, 10], 'xywh').shift((10, 0))
            <Boxes(xywh, array([35., 30., 15., 10.]))>
            >>> Boxes([25, 30, 15, 10], 'tlbr').shift((10, 5))
            <Boxes(tlbr, array([35., 35., 25., 15.]))>
        """
        boxes = self.data
        tx, ty = amount if ub.iterable(amount) else (amount, amount)
        new_data = boxes.astype(float).copy()
        if self.format in ['xywh', 'cxywh']:
            new_data[..., 0] += tx
            new_data[..., 1] += ty
        elif self.format in ['tlbr']:
            new_data[..., 0:4:2] += tx
            new_data[..., 1:4:2] += ty
        else:
            raise KeyError(self.format)
        return Boxes(new_data, self.format)

    @property
    def center(self):
        x, y, w, h = self.to_xywh()
        centerx = x + (w / 2)
        centery = y + (h / 2)
        return centerx, centery

    @property
    def shape(self):
        return self.data.shape

    @property
    def area(self):
        w, h = self.to_xywh().components[-2:]
        return w * h

    @property
    def components(self):
        a = self.data[..., 0:1]
        b = self.data[..., 1:2]
        c = self.data[..., 2:3]
        d = self.data[..., 3:4]
        return [a, b, c, d]

    @classmethod
    def _cat(cls, datas):
        return np.concatenate(datas, axis=-1)

    def toformat(self, format, copy=True):
        if format == 'xywh':
            return self.to_xywh(copy=copy)
        elif format == 'tlbr':
            return self.to_tlbr(copy=copy)
        elif format == 'cxywh':
            return self.to_cxywh(copy=copy)
        elif format == 'extent':
            return self.to_extent(copy=copy)
        else:
            raise KeyError('Cannot convert {} to {}'.format(self.format, format))

    def to_extent(self, copy=True):
        if self.format == 'extent':
            return self.copy() if copy else self
        else:
            # Only difference between tlbr and extent is the column order
            # extent is x1, x2, y1, y2
            tlbr = self.to_tlbr().data
            extent = tlbr[..., [0, 2, 1, 3]]
        return Boxes(extent, 'extent')

    def to_xywh(self, copy=True):
        if self.format == 'xywh':
            return self.copy() if copy else self
        elif self.format == 'tlbr':
            x1, y1, x2, y2 = self.components
            w = x2 - x1
            h = y2 - y1
        elif self.format == 'cxywh':
            cx, cy, w, h = self.components
            x1 = cx - w / 2
            y1 = cy - h / 2
        else:
            raise KeyError(self.format)
        xywh = self._cat([x1, y1, w, h])
        return Boxes(xywh, 'xywh')

    def to_cxywh(self, copy=True):
        if self.format == 'cxywh':
            return self.copy() if copy else self
        elif self.format == 'tlbr':
            x1, y1, x2, y2 = self.components
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
        elif self.format == 'xywh':
            x1, y1, w, h = self.components
            cx = x1 + (w / 2)
            cy = y1 + (h / 2)
        else:
            raise KeyError(self.format)
        cxywh = self._cat([cx, cy, w, h])
        return Boxes(cxywh, 'cxywh')

    def to_tlbr(self, copy=True):
        if self.format == 'tlbr':
            return self.copy() if copy else self
        elif self.format == 'cxywh':
            cx, cy, w, h = self.components
            half_w = (w / 2)
            half_h = (h / 2)
            x1 = cx - half_w
            x2 = cx + half_w
            y1 = cy - half_h
            y2 = cy + half_h
        elif self.format == 'xywh':
            x1, y1, w, h = self.components
            x2 = x1 + w
            y2 = y1 + h
        else:
            raise KeyError(self.format)
        tlbr = self._cat([x1, y1, x2, y2])
        return Boxes(tlbr, 'tlbr')

    def clip(self, x_min, y_min, x_max, y_max, inplace=False):
        """
        Clip boxes to image boundaries.  If box is in tlbr format, inplace
        operation is an option.

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> boxes = Boxes(np.array([[-10, -10, 120, 120], [1, -2, 30, 50]]), 'tlbr')
            >>> clipped = boxes.clip(0, 0, 110, 100, inplace=False)
            >>> assert np.any(boxes.data != clipped.data)
            >>> clipped2 = boxes.clip(0, 0, 110, 100, inplace=True)
            >>> assert clipped2.data is boxes.data
            >>> assert np.all(clipped2.data == clipped.data)
            >>> print(clipped)
            <Boxes(tlbr,
                array([[  0,   0, 110, 100],
                       [  1,   0,  30,  50]]))>
        """
        if inplace:
            if self.format != 'tlbr':
                raise ValueError('Must be in tlbr format to operate inplace')
            self2 = self
        else:
            self2 = self.to_tlbr(copy=True)
        x1, y1, x2, y2 = self2.data.T
        np.clip(x1, x_min, x_max, out=x1)
        np.clip(y1, y_min, y_max, out=y1)
        np.clip(x2, x_min, x_max, out=x2)
        np.clip(y2, y_min, y_max, out=y2)
        return self2

    def transpose(self):
        x, y, w, h = self.to_xywh().components
        self2 = self.__class__(self._cat([y, x, h, w]), format='xywh')
        self2 = self2.toformat(self.format)
        return self2

    def compress(self, flags, axis=0, inplace=False):
        """
        Filters boxes based on a boolean criterion

        Example:
            >>> self = Boxes([[25, 30, 15, 10]], 'tlbr')
            >>> flags = [False]
        """
        if len(self.data.shape) != 2:
            raise ValueError('data must be 2d got {}d'.format(len(self.data.shape)))
        self2 = self if inplace else self.copy()
        self2.data = self2.data.compress(flags, axis=axis)
        return self2


if __name__ == '__main__':
    """
    CommandLine:
        python -m graphid.util.util_boxes all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
