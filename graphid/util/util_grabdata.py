import ubelt as ub
from os.path import exists  # NOQA

TESTIMG_URL_DICT = {
    'astro.png' : 'https://i.imgur.com/KXhKM72.png',  # Use instead of
    'carl.jpg'  : 'http://i.imgur.com/flTHWFD.jpg',
    'grace.jpg' : 'http://i.imgur.com/rgQyu7r.jpg',
    'jeff.png'  : 'http://i.imgur.com/l00rECD.png',
    'ada2.jpg'  : 'http://i.imgur.com/zHOpTCb.jpg',
    'ada.jpg'   : 'http://i.imgur.com/iXNf4Me.jpg',
    'easy1.png' : 'http://i.imgur.com/Qqd0VNq.png',
    'easy2.png' : 'http://i.imgur.com/BDP8MIu.png',
    'easy3.png' : 'http://i.imgur.com/zBcm5mS.png',
    'hard3.png' : 'http://i.imgur.com/ST91yBf.png',
    'zebra.png' : 'http://i.imgur.com/58hbGcd.png',
    'star.png'  : 'http://i.imgur.com/d2FHuIU.png',
    'patsy.jpg' : 'http://i.imgur.com/C1lNRfT.jpg',
}


def grab_test_imgpath(key='astro.png', allow_external=True, verbose=True):
    """
    Gets paths to standard / fun test images.
    Downloads them if they dont exits

    Args:
        key (str): one of the standard test images, e.g. astro.png, carl.jpg, ...
        allow_external (bool): if True you can specify existing fpaths

    Returns:
        str: testimg_fpath - filepath to the downloaded or cached test image.

    Example:
        >>> testimg_fpath = grab_test_imgpath('carl.jpg')
        >>> assert exists(testimg_fpath)
    """
    if allow_external and key not in TESTIMG_URL_DICT:
        testimg_fpath = key
        if not exists(testimg_fpath):
            raise AssertionError(
                'testimg_fpath={!r} not found did you mean on of {!r}' % (
                    testimg_fpath, sorted(TESTIMG_URL_DICT.keys())))
    else:
        testimg_fname = key
        testimg_url = TESTIMG_URL_DICT[key]
        testimg_fpath = ub.grabdata(testimg_url, fname=testimg_fname, verbose=verbose)
    return testimg_fpath


if __name__ == '__main__':
    """
    CommandLine:
        python -m graphid.util.util_grabdata all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
