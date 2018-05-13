# -*- coding: utf-8 -*-
# LICENCE
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip
import warnings
import ubelt as ub  # NOQA
import numpy as np
import cv2
from numpy.core.umath_tests import matrix_multiply  # NOQA


def bboxes_from_vert_list(verts_list, castint=False):
    """ Fit the bounding polygon inside a rectangle """
    return [bbox_from_verts(verts, castint=castint) for verts in verts_list]


def verts_list_from_bboxes_list(bboxes_list):
    """ Create a four-vertex polygon from the bounding rectangle """
    return [verts_from_bbox(bbox) for bbox in bboxes_list]


#def mask_from_verts(verts):
#    h, w = shape[0:2]
#    y, x = np.mgrid[:h, :w]
#    points = np.transpose((x.ravel(), y.ravel()))
#    #mask = nxutils.points_inside_poly(points, verts)
#    mask = _nxutils_points_inside_poly(points, verts)
#    return mask.reshape(h, w)


def verts_from_bbox(bbox, close=False):
    """
    Args:
        bbox (tuple):  bounding box in the format (x, y, w, h)
        close (bool): (default = False)

    Returns:
        list: verts

    Example:
        >>> bbox = (10, 10, 50, 50)
        >>> close = False
        >>> verts = verts_from_bbox(bbox, close)
        >>> result = ('verts = %s' % (str(verts),))
        >>> print(result)
        verts = ((10, 10), (60, 10), (60, 60), (10, 60))
    """
    x1, y1, w, h = bbox
    x2 = (x1 + w)
    y2 = (y1 + h)
    if close:
        # Close the verticies list (for drawing lines)
        verts = ((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1))
    else:
        verts = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    return verts


def bbox_from_verts(verts, castint=False):
    x = min(x[0] for x in verts)
    y = min(y[1] for y in verts)
    w = max(x[0] for x in verts) - x
    h = max(y[1] for y in verts) - y
    if castint:
        return (int(x), int(y), int(w), int(h))
    else:
        return (x, y, w, h)


def draw_border(img_in, color=(0, 128, 255), thickness=2, out=None):
    """
    Args:
        img_in (ndarray[uint8_t, ndim=2]):  image data
        color (tuple): in bgr
        thickness (int):
        out (None):

    Example:
        >>> from graphid import util
        >>> img_in = util.imread(util.grab_test_imgpath('carl.jpg'))
        >>> color = (0, 128, 255)
        >>> thickness = 20
        >>> out = None
        >>> img = draw_border(img_in, color, thickness, out)
        >>> # xdoc: +REQUIRES(--show)
        >>> util.imshow(img)
        >>> pt.show_if_requested()
    """
    h, w = img_in.shape[0:2]
    #verts = verts_from_bbox((0, 0, w, h))
    #verts = verts_from_bbox((0, 0, w - 1, h - 1))
    half_thickness = thickness // 2
    verts = verts_from_bbox((half_thickness, half_thickness,
                             w - thickness, h - thickness))
    # FIXME: adjust verts and draw lines here to fill in the corners correctly
    img = draw_verts(img_in, verts, color=color, thickness=thickness, out=out)
    return img


def draw_verts(img_in, verts, color=(0, 128, 255), thickness=2, out=None):
    """
    Args:
        img_in (?):
        verts (?):
        color (tuple):
        thickness (int):

    Returns:
        ndarray[uint8_t, ndim=2]: img -  image data

    References:
        http://docs.opencv.org/modules/core/doc/drawing_functions.html#line

    Example:
        >>> from graphid import util
        >>> img_in = util.imread(util.grab_test_imgpath('carl.jpg'))
        >>> verts = ((10, 10), (10, 100), (100, 100), (100, 10))
        >>> color = (0, 128, 255)
        >>> thickness = 2
        >>> out = None
        >>> img = draw_verts(img_in, verts, color, thickness, out)
        >>> assert img_in is not img
        >>> assert out is not img
        >>> assert out is not img_in
        >>> # xdoc: +REQUIRES(--show)
        >>> util.imshow(img)
        >>> util.show_if_requested()

    Example1:
        >>> from graphid import util
        >>> img_in = util.imread(util.grab_test_imgpath('carl.jpg'))
        >>> verts = ((10, 10), (10, 100), (100, 100), (100, 10))
        >>> color = (0, 128, 255)
        >>> thickness = 2
        >>> out = img_in
        >>> img = draw_verts(img_in, verts, color, thickness, out)
        >>> assert img_in is img, 'should be in place'
        >>> assert out is img, 'should be in place'
        >>> # xdoc: +REQUIRES(--show)
        >>> pt.imshow(img)
        >>> pt.show_if_requested()


    out = img_in = np.zeros((500, 500, 3), dtype=np.uint8)
    """
    if out is None:
        out = np.copy(img_in)
    if isinstance(verts, np.ndarray):
        verts = verts.tolist()
    connect = True
    if connect:
        line_list_sequence = zip(verts[:-1], verts[1:])
        line_tuple_sequence = ((tuple(p1_), tuple(p2_)) for (p1_, p2_) in line_list_sequence)
        cv2.line(out, tuple(verts[0]), tuple(verts[-1]), color, thickness)
        for (p1, p2) in line_tuple_sequence:
            cv2.line(out, p1, p2, color, thickness)
            #print('p1, p2: (%r, %r)' % (p1, p2))
    else:
        for count, p in enumerate(verts, start=1):
            cv2.circle(out, tuple(p), count, color, thickness=1)
    return out


def closest_point_on_line_segment(p, e1, e2):
    """
    Finds the closet point from p on line segment (e1, e2)

    Args:
        p (ndarray): and xy point
        e1 (ndarray): the first xy endpoint of the segment
        e2 (ndarray): the second xy endpoint of the segment

    Returns:
        ndarray: pt_on_seg - the closest xy point on (e1, e2) from p

    References:
        http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        http://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment

    Example:
        >>> verts = np.array([[ 21.83012702,  13.16987298],
        >>>                   [ 16.83012702,  21.83012702],
        >>>                   [  8.16987298,  16.83012702],
        >>>                   [ 13.16987298,   8.16987298],
        >>>                   [ 21.83012702,  13.16987298]])
        >>> rng = np.random.RandomState(0)
        >>> p_list = rng.rand(64, 2) * 20 + 5
        >>> close_pts = np.array([closest_point_on_vert_segments(p, verts) for p in p_list])
        >>> from graphid import util
        >>> import matplotlib.pyplot as plt
        >>> util.qtensure()
        >>> plt.plot(p_list.T[0], p_list.T[1], 'ro', label='original point')
        >>> plt.plot(close_pts.T[0], close_pts.T[1], 'rx', label='closest point on shape')
        >>> for x, y in list(zip(p_list, close_pts)):
        >>>     z = np.array(list(zip(x, y)))
        >>>     plt.plot(z[0], z[1], 'r--')
        >>> plt.legend()
        >>> plt.plot(verts.T[0], verts.T[1], 'b-')
        >>> plt.xlim(0, 30)
        >>> plt.ylim(0, 30)
        >>> plt.axis('equal')
        >>> util.show_if_requested()
    """
    # shift e1 to origin
    de = (dx, dy) = e2 - e1
    # make point vector wrt orgin
    pv = p - e1
    # Project pv onto de
    mag = np.linalg.norm(de)
    pt_on_line_ = pv.dot(de / mag) * de / mag
    # Check if normalized dot product is between 0 and 1
    # Determines if pt is between 0,0 and de
    t = de.dot(pt_on_line_) / mag ** 2
    # t is an interpolation factor indicating how far past the line segment we
    # are. We are on the line segment if it is in the range 0 to 1.
    if t < 0:
        pt_on_seg = e1
    elif t > 1:
        pt_on_seg = e2
    else:
        pt_on_seg = pt_on_line_ + e1
    return pt_on_seg


def distance_to_lineseg(p, e1, e2):
    from graphid import util
    close_pt = closest_point_on_line_segment(p, e1, e2)
    dist_to_lineseg = util.L2(p, close_pt)
    return dist_to_lineseg


def closest_point_on_line(p, e1, e2):
    """
    e1 and e2 define two points on the line.
    Does not clip to the segment.

    Example:
        >>> verts = np.array([[ 21.83012702,  13.16987298],
        >>>                   [ 16.83012702,  21.83012702],
        >>>                   [  8.16987298,  16.83012702],
        >>>                   [ 13.16987298,   8.16987298],
        >>>                   [ 21.83012702,  13.16987298]])
        >>> rng = np.random.RandomState(0)
        >>> p_list = rng.rand(64, 2) * 20 + 5
        >>> close_pts = []
        >>> for p in p_list:
        >>>     candidates = [closest_point_on_line(p, e1, e2) for e1, e2 in ub.iter_window(verts, 2)]
        >>>     dists = np.array([np.linalg.norm(p - new_pt, axis=0) for new_pt in candidates])
        >>>     close_pts.append(candidates[dists.argmin()])
        >>> close_pts = np.array(close_pts)
        >>> from graphid import util
        >>> import matplotlib.pyplot as plt
        >>> util.qtensure()
        >>> plt.plot(p_list.T[0], p_list.T[1], 'ro', label='original point')
        >>> plt.plot(close_pts.T[0], close_pts.T[1], 'rx', label='closest point on shape')
        >>> for x, y in list(zip(p_list, close_pts)):
        >>>     z = np.array(list(zip(x, y)))
        >>>     plt.plot(z[0], z[1], 'r--')
        >>> plt.legend()
        >>> plt.plot(verts.T[0], verts.T[1], 'b-')
        >>> plt.xlim(0, 30)
        >>> plt.ylim(0, 30)
        >>> plt.axis('equal')
        >>> util.show_if_requested()
    """
    # shift e1 to origin
    de = (dx, dy) = e2 - e1
    # make point vector wrt orgin
    pv = p - e1
    # Project pv onto de
    mag = np.linalg.norm(de)
    pt_on_line_ = pv.dot(de / mag) * de / mag
    pt_on_line = pt_on_line_ + e1
    return pt_on_line


def closest_point_on_vert_segments(p, verts):
    candidates = [closest_point_on_line_segment(p, e1, e2) for e1, e2 in ub.iter_window(verts, 2)]
    dists = np.array([np.linalg.norm(p - new_pt, axis=0) for new_pt in candidates])
    new_pts = candidates[dists.argmin()]
    return new_pts


def closest_point_on_bbox(p, bbox):
    """

    Example1:
        >>> p_list = np.array([[19, 7], [7, 14], [14, 11], [8, 7], [23, 21]], dtype=np.float)
        >>> bbox = np.array([10, 10, 10, 10], dtype=np.float)
        >>> [closest_point_on_bbox(p, bbox) for p in p_list]
    """
    verts = np.array(verts_from_bbox(bbox, close=True))
    new_pts = closest_point_on_vert_segments(p, verts)
    return new_pts


def bbox_from_xywh(xy, wh, xy_rel_pos=[0, 0]):
    """ need to specify xy_rel_pos if xy is not in tl already """
    to_tlx = xy_rel_pos[0] * wh[0]
    to_tly = xy_rel_pos[1] * wh[1]
    tl_x = xy[0] - to_tlx
    tl_y = xy[1] - to_tly
    bbox = [tl_x, tl_y, wh[0], wh[1]]
    return bbox


def extent_from_verts(verts):
    bbox = bbox_from_verts(verts)
    extent = extent_from_bbox(bbox)
    return extent


def union_extents(extents):
    extents = np.array(extents)
    xmin = extents.T[0].min()
    xmax = extents.T[1].max()
    ymin = extents.T[2].min()
    ymax = extents.T[3].max()
    return (xmin, xmax, ymin, ymax)


#def tlbr_from_bbox(bbox):
def extent_from_bbox(bbox):
    """
    Args:
        bbox (ndarray): tl_x, tl_y, w, h

    Returns:
        extent (ndarray): tl_x, br_x, tl_y, br_y

    Example:
        >>> bbox = [0, 0, 10, 10]
        >>> extent = extent_from_bbox(bbox)
        >>> result = ('extent = %s' % (ub.repr2(list(extent), nl=0),))
        >>> print(result)
        extent = [0, 10, 0, 10]
    """
    tl_x, tl_y, w, h = bbox
    br_x = tl_x + w
    br_y = tl_y + h
    extent = [tl_x, br_x, tl_y, br_y]
    return extent


#def tlbr_from_bbox(bbox):
def bbox_from_extent(extent):
    """
    Args:
        extent (ndarray): tl_x, br_x, tl_y, br_y

    Returns:
        bbox (ndarray): tl_x, tl_y, w, h

    Example:
        >>> extent = [0, 10, 0, 10]
        >>> bbox = bbox_from_extent(extent)
        >>> print('bbox = {}'.format(ub.repr2(list(bbox), nl=0)))
        bbox = [0, 0, 10, 10]
    """
    tl_x, br_x, tl_y, br_y = extent
    w = br_x - tl_x
    h = br_y - tl_y
    bbox = [tl_x, tl_y, w, h]
    return bbox


def bbox_from_center_wh(center_xy, wh):
    return bbox_from_xywh(center_xy, wh, xy_rel_pos=[.5, .5])


def bbox_center(bbox):
    from graphid import util
    return util.Boxes(bbox).center
    # (x, y, w, h) = bbox
    # centerx = x + (w / 2)
    # centery = y + (h / 2)
    # return centerx, centery


def get_pointset_extents(pts):
    minx, miny = pts.min(axis=0)
    maxx, maxy = pts.max(axis=0)
    bounds = minx, maxx, miny, maxy
    return bounds


def get_pointset_extent_wh(pts):
    minx, miny = pts.min(axis=0)
    maxx, maxy = pts.max(axis=0)
    extent_w = maxx - minx
    extent_h = maxy - miny
    return extent_w, extent_h


def cvt_bbox_xywh_to_pt1pt2(xywh, sx=1.0, sy=1.0, round_=True):
    """ Converts bbox to thumb format with a scale facto"""
    from graphid import util
    (x1, y1, _w, _h) = xywh
    x2 = (x1 + _w)
    y2 = (y1 + _h)
    if round_:
        pt1 = (util.iround(x1 * sx), util.iround(y1 * sy))
        pt2 = (util.iround(x2 * sx), util.iround(y2 * sy))
    else:
        pt1 = ((x1 * sx), (y1 * sy))
        pt2 = ((x2 * sx), (y2 * sy))
    return (pt1, pt2)


def scale_bbox(bbox, sx, sy=None):
    if sy is None:
        sy = sx
    centerx, centery = bbox_center(bbox)
    S = scale_around_mat3x3(sx, sy, centerx, centery)
    verts = np.array(verts_from_bbox(bbox))
    vertsT = transform_points_with_homography(S, verts.T).T
    bboxT = bbox_from_verts(vertsT)
    return bboxT


def scale_around_mat3x3(sx, sy, x, y):
    scale_ = scale_mat3x3(sx, sy)
    return transform_around(scale_, x, y)


TRANSFORM_DTYPE = np.float64


def translation_mat3x3(x, y, dtype=TRANSFORM_DTYPE):
    T = np.array([[1, 0,  x],
                  [0, 1,  y],
                  [0, 0,  1]], dtype=dtype)
    return T


def transform_around(M, x, y):
    """ translates to origin, applies transform and then translates back """
    tr1_ = translation_mat3x3(-x, -y)
    tr2_ = translation_mat3x3(x, y)
    M_ = tr2_.dot(M).dot(tr1_)
    return M_


def scale_mat3x3(sx, sy=None, dtype=TRANSFORM_DTYPE):
    sy = sx if sy is None else sy
    S = np.array([[sx, 0, 0],
                  [0, sy, 0],
                  [0,  0, 1]], dtype=dtype)
    return S


def transform_points_with_homography(H, _xys):
    """
    Args:
        H (ndarray[float64_t, ndim=2]):  homography/perspective matrix
        _xys (ndarray[ndim=2]): (2 x N) array
    """
    xyz  = add_homogenous_coordinate(_xys)
    xyz_t = matrix_multiply(H, xyz)
    xy_t  = remove_homogenous_coordinate(xyz_t)
    return xy_t


def scale_extents(extents, sx, sy=None):
    """
    Args:
        extent (ndarray): tl_x, br_x, tl_y, br_y
    """
    bbox = bbox_from_extent(extents)
    bboxT = scale_bbox(bbox, sx, sy)
    extentsT = extent_from_bbox(bboxT)
    return extentsT


def scaled_verts_from_bbox_gen(bbox_list, theta_list, sx=1, sy=1):
    """
    Helps with drawing scaled bbounding boxes on thumbnails

    Args:
        bbox_list (list): bboxes in x,y,w,h format
        theta_list (list): rotation of bounding boxes
        sx (float): x scale factor
        sy (float): y scale factor

    Yeilds:
        new_verts - vertices of scaled bounding box for every input

    Example:
        >>> bbox_list = [(10, 10, 100, 100)]
        >>> theta_list = [0]
        >>> sx = .5
        >>> sy = .5
        >>> new_verts_list = list(scaled_verts_from_bbox_gen(bbox_list, theta_list, sx, sy))
        >>> result = str(new_verts_list)
        >>> print(result)
        [[[5, 5], [55, 5], [55, 55], [5, 55], [5, 5]]]
    """
    # TODO: input verts support and better name
    for bbox, theta in zip(bbox_list, theta_list):
        new_verts = scaled_verts_from_bbox(bbox, theta, sx, sy)
        yield new_verts


def scaled_verts_from_bbox(bbox, theta, sx, sy):
    """
    Helps with drawing scaled bbounding boxes on thumbnails

    """
    if bbox is None:
        return None
    # Transformation matrixes
    R = rotation_around_bbox_mat3x3(theta, bbox)
    S = scale_mat3x3(sx, sy)
    # Get verticies of the annotation polygon
    verts = verts_from_bbox(bbox, close=True)
    # Rotate and transform to thumbnail space
    xyz_pts = add_homogenous_coordinate(np.array(verts).T)
    trans_pts = remove_homogenous_coordinate(S.dot(R).dot(xyz_pts))
    new_verts = np.round(trans_pts).astype(np.int32).T.tolist()
    return new_verts


def point_inside_bbox(point, bbox):
    """
    Flags points that are strictly inside a bounding box.
    Points on the boundary are not considered inside.

    Args:
        point (ndarray): one or more points to test (2xN)
        bbox (tuple): a bounding box in  (x, y, w, h) format

    Returns:
        bool or ndarray: True if the point is in the bbox

    Example:
        >>> point = np.array([
        >>>     [3, 2], [4, 1], [2, 3], [1, 1], [0, 0],
        >>>     [4, 9.5], [9, 9.5], [7, 2], [7, 8], [9, 3]
        >>> ]).T
        >>> bbox = (3, 2, 5, 7)
        >>> flag = point_inside_bbox(point, bbox)
        >>> flag = flag.astype(np.int)
        >>> result = ('flag = %s' % (ub.repr2(flag),))
        >>> print(result)
        >>> # xdoc: +REQUIRES(--show)
        >>> verts = np.array(verts_from_bbox(bbox, close=True))
        >>> util.plot(verts.T[0], verts.T[1], 'b-')
        >>> util.plot(point[0][flag], point[1][flag], 'go')
        >>> util.plot(point[0][~flag], point[1][~flag], 'rx')
        >>> plt.xlim(0, 10); plt.ylim(0, 10)
        >>> from graphid import util
        >>> util.show_if_requested()
        flag = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    """
    x, y = point
    tl_x, br_x, tl_y, br_y = extent_from_bbox(bbox)
    inside_x = np.logical_and(tl_x < x, x < br_x)
    inside_y = np.logical_and(tl_y < y, y < br_y)
    flag = np.logical_and(inside_x, inside_y)
    return flag


def add_homogenous_coordinate(_xys):
    """
    Example:
        >>> _xys = np.array([[ 2.,  0.,  0.,  2.],
        ...                  [ 2.,  2.,  0.,  0.]], dtype=np.float32)
        >>> _xyzs = add_homogenous_coordinate(_xys)
        >>> assert np.all(_xys == remove_homogenous_coordinate(_xyzs))
        >>> result = ub.repr2(_xyzs, with_dtype=True)
        >>> print(result)
        np.array([[ 2.,  0.,  0.,  2.],
                  [ 2.,  2.,  0.,  0.],
                  [ 1.,  1.,  1.,  1.]], dtype=np.float32)
    """
    assert _xys.shape[0] == 2
    _zs = np.ones((1, _xys.shape[1]), dtype=_xys.dtype)
    _xyzs = np.vstack((_xys, _zs))
    return _xyzs


def remove_homogenous_coordinate(_xyzs):
    """
    normalizes 3d homogonous coordinates into 2d coordinates

    Args:
        _xyzs (ndarray): of shape (3, N)

    Returns:
        ndarray: _xys of shape (2, N)

    Example:
        >>> _xyzs = np.array([[ 2.,   0.,  0.,  2.],
        ...                   [ 2.,   2.,  0.,  0.],
        ...                   [ 1.2,  1.,  1.,  2.]], dtype=np.float32)
        >>> _xys = remove_homogenous_coordinate(_xyzs)
        >>> print(ub.repr2(_xys, precision=3, with_dtype=True))
        np.array([[ 1.667,  0.   ,  0.   ,  1.   ],
                  [ 1.667,  2.   ,  0.   ,  0.   ]], dtype=np.float32)

    Example:
        >>> _xyzs = np.array([[ 140.,  167.,  185.,  185.,  194.],
        ...                   [ 121.,  139.,  156.,  155.,  163.],
        ...                   [  47.,   56.,   62.,   62.,   65.]])
        >>> _xys = remove_homogenous_coordinate(_xyzs)
        >>> print(ub.repr2(_xys, precision=3, with_dtype=False))
        np.array([[ 2.979,  2.982,  2.984,  2.984,  2.985],
                  [ 2.574,  2.482,  2.516,  2.5  ,  2.508]])
    """
    assert _xyzs.shape[0] == 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _xys = np.divide(_xyzs[0:2], _xyzs[None, 2])
    return _xys


def rotation_around_bbox_mat3x3(theta, bbox):
    x, y, w, h = bbox
    centerx = x + (w / 2)
    centery = y + (h / 2)
    return rotation_around_mat3x3(theta, centerx, centery)


def rotation_around_mat3x3(theta, x, y):
    # rot = rotation_mat3x3(theta)
    # return transform_around(rot, x, y)
    tr1_ = translation_mat3x3(-x, -y)
    rot_ = rotation_mat3x3(theta)
    tr2_ = translation_mat3x3(x, y)
    rot = tr2_.dot(rot_).dot(tr1_)
    return rot


def rotation_mat3x3(radians, sin=np.sin, cos=np.cos):
    """
    References:
        https://en.wikipedia.org/wiki/Rotation_matrix
    """
    # TODO: handle array inputs
    sin_ = sin(radians)
    cos_ = cos(radians)
    R = np.array(((cos_, -sin_,  0),
                  (sin_,  cos_,  0),
                  (   0,     0,  1),))
    return R


if __name__ == '__main__':
    """
    CommandLine:
        python -m graphid.util.util_geometry all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
