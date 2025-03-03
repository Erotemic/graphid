from importlib import resources as importlib_resources
import ubelt as ub


def requirement_path(fname):
    """
    CommandLine:
        xdoctest -m graphid.rc.registry requirement_path

    Example:
        >>> from graphid.rc.registry import requirement_path
        >>> fname = 'runtime.txt'
        >>> fpath = requirement_path(fname)
        >>> print(fpath)
    """
    with importlib_resources.path('graphid.rc.requirements', f'{fname}') as p:
        orig_pth = ub.Path(p)
        return orig_pth
