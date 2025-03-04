import warnings
import numpy as np
import cv2
try:
    import skimage.io
except ImportError:
    pass


def ensure_float01(img, dtype=np.float32, copy=True):
    """ Ensure that an image is encoded using a float properly """
    if img.dtype.kind in ('i', 'u'):
        assert img.max() <= 255
        img_ = img.astype(dtype, copy=copy) / 255.0
    else:
        img_ = img.astype(dtype, copy=copy)
    return img_


def get_num_channels(img):
    """ Returns the number of color channels """
    ndims = len(img.shape)
    if ndims == 2:
        nChannels = 1
    elif ndims == 3 and img.shape[2] == 3:
        nChannels = 3
    elif ndims == 3 and img.shape[2] == 4:
        nChannels = 4
    elif ndims == 3 and img.shape[2] == 1:
        nChannels = 1
    else:
        raise ValueError('Cannot determine number of channels '
                         'for img.shape={}'.format(img.shape))
    return nChannels


def convert_colorspace(img, dst_space, src_space='BGR', copy=False, dst=None):
    """
    Converts colorspace of img.
    Convinience function around cv2.cvtColor

    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        colorspace (str): RGB, LAB, etc
        dst_space (unicode): (default = u'BGR')

    Returns:
        ndarray[uint8_t, ndim=2]: img -  image data

    Example:
        >>> # xdoctest: +SKIP("use kwimage instead")
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> convert_colorspace(np.array([[[0, 0, 1]]], dtype=np.float32), 'LAB', src_space='RGB')
        >>> convert_colorspace(np.array([[[0, 1, 0]]], dtype=np.float32), 'LAB', src_space='RGB')
        >>> convert_colorspace(np.array([[[1, 0, 0]]], dtype=np.float32), 'LAB', src_space='RGB')
        >>> convert_colorspace(np.array([[[1, 1, 1]]], dtype=np.float32), 'LAB', src_space='RGB')
        >>> convert_colorspace(np.array([[[0, 0, 1]]], dtype=np.float32), 'HSV', src_space='RGB')
    """
    dst_space = dst_space.upper()
    src_space = src_space.upper()
    if src_space == dst_space:
        img2 = img
        if dst is not None:
            dst[...] = img[...]
            img2 = dst
        elif copy:
            img2 = img2.copy()
    else:
        code = _lookup_colorspace_code(dst_space, src_space)
        # Note the conversion to colorspaces like LAB and HSV in float form
        # do not go into the 0-1 range. Instead they go into
        # (0-100, -111-111hs, -111-111is) and (0-360, 0-1, 0-1) respectively
        img2 = cv2.cvtColor(img, code, dst=dst)
    return img2


def _lookup_colorspace_code(dst_space, src_space='BGR'):
    src = src_space.upper()
    dst = dst_space.upper()
    convert_attr = 'COLOR_{}2{}'.format(src, dst)
    if not hasattr(cv2, convert_attr):
        prefix = 'COLOR_{}2'.format(src)
        valid_dst_spaces = [
            key.replace(prefix, '')
            for key in cv2.__dict__.keys() if key.startswith(prefix)]
        raise KeyError(
            '{} does not exist, valid conversions from {} are to {}'.format(
                convert_attr, src_space, valid_dst_spaces))
    else:
        code = getattr(cv2, convert_attr)
    return code


def imread(fpath, **kw):
    """
    reads image data in BGR format

    Example:
        >>> # xdoctest: +SKIP("use kwimage.imread")
        >>> import ubelt as ub
        >>> import tempfile
        >>> from os.path import splitext  # NOQA
        >>> fpath = ub.grabdata('https://i.imgur.com/oHGsmvF.png', fname='carl.png')
        >>> #fpath = ub.grabdata('http://www.topcoder.com/contest/problem/UrbanMapper3D/JAX_Tile_043_DTM.tif')
        >>> ext = splitext(fpath)[1]
        >>> img1 = imread(fpath)
        >>> # Check that write + read preserves data
        >>> tmp = tempfile.NamedTemporaryFile(suffix=ext)
        >>> imwrite(tmp.name, img1)
        >>> img2 = imread(tmp.name)
        >>> assert np.all(img2 == img1)

    Example:
        >>> # xdoctest: +SKIP("use kwimage.imread")
        >>> import tempfile
        >>> import ubelt as ub
        >>> #img1 = (np.arange(0, 12 * 12 * 3).reshape(12, 12, 3) % 255).astype(np.uint8)
        >>> img1 = imread(ub.grabdata('http://i.imgur.com/iXNf4Me.png', fname='ada.png'))
        >>> tmp_tif = tempfile.NamedTemporaryFile(suffix='.tif')
        >>> tmp_png = tempfile.NamedTemporaryFile(suffix='.png')
        >>> imwrite(tmp_tif.name, img1)
        >>> imwrite(tmp_png.name, img1)
        >>> tif_im = imread(tmp_tif.name)
        >>> png_im = imread(tmp_png.name)
        >>> assert np.all(tif_im == png_im)

    Example:
        >>> # xdoctest: +SKIP("use kwimage.imread")
        >>> from graphid.util.util_image import *
        >>> import tempfile
        >>> import ubelt as ub
        >>> #img1 = (np.arange(0, 12 * 12 * 3).reshape(12, 12, 3) % 255).astype(np.uint8)
        >>> tif_fpath = ub.grabdata('https://ghostscript.com/doc/tiff/test/images/rgb-3c-16b.tiff')
        >>> img1 = imread(tif_fpath)
        >>> tmp_tif = tempfile.NamedTemporaryFile(suffix='.tif')
        >>> tmp_png = tempfile.NamedTemporaryFile(suffix='.png')
        >>> imwrite(tmp_tif.name, img1)
        >>> imwrite(tmp_png.name, img1)
        >>> tif_im = imread(tmp_tif.name)
        >>> png_im = imread(tmp_png.name)
        >>> assert np.all(tif_im == png_im)
    """
    try:
        if fpath.endswith(('.tif', '.tiff')):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # skimage reads in RGB, convert to BGR
                image = skimage.io.imread(fpath, **kw)
                if get_num_channels(image) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif get_num_channels(image) == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        else:
            image = cv2.imread(fpath, flags=cv2.IMREAD_UNCHANGED)
            if image is None:
                raise IOError('OpenCV cannot read this image')
        return image
    except Exception:
        print('Error reading fpath = {!r}'.format(fpath))
        raise


def imwrite(fpath, image, **kw):
    """
    writes image data in BGR format
    """
    if fpath.endswith(('.tif', '.tiff')):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # skimage writes in RGB, convert from BGR
            if get_num_channels(image) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif get_num_channels(image) == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            return skimage.io.imsave(fpath, image)
    else:
        return cv2.imwrite(fpath, image)


if __name__ == '__main__':
    """
    CommandLine:
        python -m graphid.util.util_image all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
