import ubelt as ub
import itertools as it
import numpy as np


def randn(mean=0, std=1, shape=[], a_max=None, a_min=None, rng=None):
    a = (rng.randn(*shape) * std) + mean
    if a_max is not None or a_min is not None:
        a = np.clip(a, a_min, a_max)
    return a


def aslist(sequence):
    """
    Ensures that the sequence object is a Python list.
    Handles, numpy arrays, and python sequences (e.g. tuples, and iterables).

    Args:
        sequence (sequence): a list-like object

    Returns:
        list: list_ - `sequence` as a Python list

    Example:
        >>> s1 = [1, 2, 3]
        >>> s2 = (1, 2, 3)
        >>> assert aslist(s1) is s1
        >>> assert aslist(s2) is not s2
        >>> aslist(np.array([[1, 2], [3, 4], [5, 6]]))
        [[1, 2], [3, 4], [5, 6]]
        >>> aslist(range(3))
        [0, 1, 2]
    """
    if isinstance(sequence, list):
        return sequence
    elif isinstance(sequence, np.ndarray):
        list_ = sequence.tolist()
    else:
        list_ = list(sequence)
    return list_


class classproperty(property):
    """
    Decorates a method turning it into a classattribute

    References:
        https://stackoverflow.com/questions/1697501/python-staticmethod-with-property
    """
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


def estarmap(func, iter_, **kwargs):
    """
    Eager version of it.starmap from itertools

    Note this is inefficient and should only be used when prototyping and
    debugging.
    """
    return [func(*arg, **kwargs) for arg in iter_]


def delete_dict_keys(dict_, key_list):
    r"""
    Removes items from a dictionary inplace. Keys that do not exist are
    ignored.

    Args:
        dict_ (dict): dict like object with a __del__ attribute
        key_list (list): list of keys that specify the items to remove

    Example:
        >>> dict_ = {'bread': 1, 'churches': 1, 'cider': 2, 'very small rocks': 2}
        >>> key_list = ['duck', 'bread', 'cider']
        >>> delete_dict_keys(dict_, key_list)
        >>> result = ub.urepr(dict_, nl=False)
        >>> print(result)
        {'churches': 1, 'very small rocks': 2}

    """
    invalid_keys = set(key_list) - set(dict_.keys())
    valid_keys = set(key_list) - invalid_keys
    for key in valid_keys:
        del dict_[key]
    return dict_


def flag_None_items(list_):
    return [item is None for item in list_]


def where(flag_list):
    """ takes flags returns indexes of True values """
    return [index for index, flag in enumerate(flag_list) if flag]


def delete_items_by_index(list_, index_list, copy=False):
    """
    Remove items from ``list_`` at positions specified in ``index_list``
    The original ``list_`` is preserved if ``copy`` is True

    Args:
        list_ (list):
        index_list (list):
        copy (bool): preserves original list if True

    Example:
        >>> list_ = [8, 1, 8, 1, 6, 6, 3, 4, 4, 5, 6]
        >>> index_list = [2, -1]
        >>> result = delete_items_by_index(list_, index_list)
        >>> print(result)
        [8, 1, 1, 6, 6, 3, 4, 4, 5]
    """
    if copy:
        list_ = list_[:]
    # Rectify negative indicies
    index_list_ = [(len(list_) + x if x < 0 else x) for x in index_list]
    # Remove largest indicies first
    index_list_ = sorted(index_list_, reverse=True)
    for index in index_list_:
        del list_[index]
    return list_


def make_index_lookup(list_, dict_factory=dict):
    r"""
    Args:
        list_ (list): assumed to have unique items

    Returns:
        dict: mapping from item to index

    Example:
        >>> list_ = [5, 3, 8, 2]
        >>> idx2_item = make_index_lookup(list_)
        >>> result = ub.urepr(idx2_item, nl=False, sort=1)
        >>> assert list(ub.take(idx2_item, list_)) == list(range(len(list_)))
        >>> print(result)
        {2: 3, 3: 1, 5: 0, 8: 2}
    """
    return dict_factory(zip(list_, range(len(list_))))


def cprint(text, color=None):
    """ provides some color to terminal output

    Args:
        text (str):
        color (str):

    Ignore:
        assert color in ['', 'yellow', 'blink', 'lightgray', 'underline',
        'darkyellow', 'blue', 'darkblue', 'faint', 'fuchsia', 'black', 'white',
        'red', 'brown', 'turquoise', 'bold', 'darkred', 'darkgreen', 'reset',
        'standout', 'darkteal', 'darkgray', 'overline', 'purple', 'green', 'teal',
        'fuscia']

    Example0:
        >>> import pygments.console
        >>> msg_list = list(pygments.console.codes.keys())
        >>> color_list = list(pygments.console.codes.keys())
        >>> [cprint(text, color) for text, color in zip(msg_list, color_list)]

    Example1:
        >>> import pygments.console
        >>> print('line1')
        >>> cprint('line2', 'red')
        >>> cprint('line3', 'blue')
        >>> cprint('line4', 'magenta')
        >>> cprint('line5', 'reset')
        >>> cprint('line5', 'magenta')
        >>> print('line6')
    """
    if False and ub.WIN32:
        # Ignore colors on windows. Seems to cause a recursion error
        print(text)
    else:
        try:
            if color is None:
                print(ub.color_text(text, 'blue'))
            else:
                print(ub.color_text(text, color))
        except RecursionError:
            print(text)


def ensure_iterable(obj):
    """
    Args:
        obj (scalar or iterable):

    Returns:
        it3erable: obj if it was iterable otherwise [obj]

    Timeit:
        %timeit util.ensure_iterable([1])
        %timeit util.ensure_iterable(1)
        %timeit util.ensure_iterable(np.array(1))
        %timeit util.ensure_iterable([1])
        %timeit [1]


    Example:
        >>> obj_list = [3, [3], '3', (3,), [3,4,5]]
        >>> result = [ensure_iterable(obj) for obj in obj_list]
        >>> result = str(result)
        >>> print(result)
        [[3], [3], ['3'], (3,), [3, 4, 5]]
    """
    if ub.iterable(obj):
        return obj
    else:
        return [obj]


def highlight_regex(str_, pat, reflags=0, color='red'):
    """
    FIXME Use pygments instead
    """
    import re
    matches = list(re.finditer(pat, str_, flags=reflags))
    colored = str_
    for match in reversed(matches):
        start = match.start()
        end = match.end()
        colored_part = ub.color_text(colored[start:end], color)
        colored = colored[:start] + colored_part + colored[end:]
    return colored


def regex_word(w):
    return r'\b%s\b' % (w,)


def setdiff(list1, list2):
    """
    returns list1 elements that are not in list2. preserves order of list1

    Args:
        list1 (list):
        list2 (list):

    Returns:
        list: new_list

    Example:
        >>> list1 = ['featweight_rowid', 'feature_rowid', 'config_rowid', 'featweight_forground_weight']
        >>> list2 = [u'featweight_rowid']
        >>> new_list = setdiff(list1, list2)
        >>> result = ub.urepr(new_list, nl=False)
        >>> print(result)
        ['feature_rowid', 'config_rowid', 'featweight_forground_weight']
    """
    set2 = set(list2)
    return [item for item in list1 if item not in set2]


def all_dict_combinations(varied_dict):
    """
    all_dict_combinations

    Args:
        varied_dict (dict):  a dict with lists of possible parameter settings

    Returns:
        list: dict_list a list of dicts correpsonding to all combinations of params settings

    Example:
        >>> varied_dict = {'logdist_weight': [0.0, 1.0], 'pipeline_root': ['vsmany'], 'sv_on': [True, False, None]}
        >>> dict_list = all_dict_combinations(varied_dict)
        >>> result = str(ub.urepr(dict_list))
        >>> print(result)
        [
            {'logdist_weight': 0.0, 'pipeline_root': 'vsmany', 'sv_on': True},
            {'logdist_weight': 0.0, 'pipeline_root': 'vsmany', 'sv_on': False},
            {'logdist_weight': 0.0, 'pipeline_root': 'vsmany', 'sv_on': None},
            {'logdist_weight': 1.0, 'pipeline_root': 'vsmany', 'sv_on': True},
            {'logdist_weight': 1.0, 'pipeline_root': 'vsmany', 'sv_on': False},
            {'logdist_weight': 1.0, 'pipeline_root': 'vsmany', 'sv_on': None},
        ]
    """
    tups_list = [[(key, val) for val in val_list]
                 if isinstance(val_list, (list))
                 else [(key, val_list)]
                 for (key, val_list) in iteritems_sorted(varied_dict)]
    dict_list = [dict(tups) for tups in it.product(*tups_list)]
    return dict_list


def iteritems_sorted(dict_):
    """ change to iteritems ordered """
    if isinstance(dict_, ub.odict):
        return dict_.items()
    else:
        return iter(sorted(dict_.items()))


def partial_order(list_, part):
    list_items = set(list_)
    part_items = set(part)
    begin = [p for p in part if p in list_items]
    end = [item for item in list_ if item not in part_items]
    return begin + end


def replace_nones(list_, repl=-1):
    r"""
    Recursively removes Nones in all lists and sublists and replaces them with
    the repl variable

    Args:
        list_ (list):
        repl (obj): replacement value

    Returns:
        list

    Example:
        >>> list_ = [None, 0, 1, 2]
        >>> repl = -1
        >>> repl_list = replace_nones(list_, repl)
        >>> result = str(repl_list)
        >>> print(result)
        [-1, 0, 1, 2]

    """
    repl_list = [
        repl if item is None else (
            replace_nones(item, repl) if isinstance(item, list) else item
        )
        for item in list_
    ]
    return repl_list


def take_percentile_parts(arr, front=None, mid=None, back=None):
    """
    Take parts from front, back, or middle of a list

    Example:
        >>> arr = list(range(20))
        >>> front = 3
        >>> mid = 3
        >>> back = 3
        >>> result = take_percentile_parts(arr, front, mid, back)
        >>> print(result)
        [0, 1, 2, 9, 10, 11, 17, 18, 19]
    """
    slices = []
    if front:
        slices += [snapped_slice(len(arr), 0.0, front)]
    if mid:
        slices += [snapped_slice(len(arr), 0.5, mid)]
    if back:
        slices += [snapped_slice(len(arr), 1.0, back)]
    parts = list(ub.flatten([arr[sl] for sl in slices]))
    return parts


def snapped_slice(size, frac, n):
    r"""
    Creates a slice spanning `n` items in a list of length `size` at position
    `frac`.

    Args:
        size (int): length of the list
        frac (float): position in the range [0, 1]
        n (int): number of items in the slice

    Returns:
        slice: slice object that best fits the criteria

    SeeAlso:
        take_percentile_parts

    Example:

    Example:
        >>> # DISABLE_DOCTEST
        >>> print(snapped_slice(0, 0, 10))
        >>> print(snapped_slice(1, 0, 10))
        >>> print(snapped_slice(100, 0, 10))
        >>> print(snapped_slice(9, 0, 10))
        >>> print(snapped_slice(100, 1, 10))
        pass
    """
    from math import floor, ceil
    if size < n:
        n = size
    start = int(size * frac - ceil(n / 2)) + 1
    stop  = int(size * frac + floor(n / 2)) + 1
    # slide to the front or the back
    buf = 0
    if stop >= size:
        buf = (size - stop)
    elif start < 0:
        buf = 0 - start
    stop += buf
    start += buf
    assert stop <= size, 'out of bounds [%r, %r]' % (stop, start)
    sl = slice(start, stop)
    return sl


def get_timestamp(format_='iso', use_second=False, delta_seconds=None,
                  isutc=False, timezone=False):
    """
    get_timestamp

    Args:
        format_ (str): (tag, printable, filename, other)
        use_second (bool):
        delta_seconds (None):

    Returns:
        str: stamp

    Example:
        >>> format_ = 'printable'
        >>> use_second = False
        >>> delta_seconds = None
        >>> stamp = get_timestamp(format_, use_second, delta_seconds)
        >>> print(stamp)
        >>> assert len(stamp) == len('15:43:04 2015/02/24')
    """
    # TODO: time.timezone
    import time
    import datetime
    if format_ == 'int':
        if isutc:
            stamp = int(time.mktime(time.gmtime()))
        else:
            stamp = int(time.mktime(time.localtime()))
        return stamp
    if isutc:
        now = datetime.datetime.utcnow()
    else:
        now = datetime.datetime.now()
    if delta_seconds is not None:
        now += datetime.timedelta(seconds=delta_seconds)
    if format_ == 'iso':
        # ISO 8601
        #utcnow = datetime.datetime.utcnow()
        #utcnow.isoformat()
        localOffsetHour = time.timezone // 3600
        utc_offset = '-' + str(localOffsetHour) if localOffsetHour < 0 else '+' + str(localOffsetHour)
        stamp = time.strftime('%Y-%m-%dT%H%M%S') + utc_offset
        return stamp
    if format_ == 'tag':
        time_tup = (now.year - 2000, now.month, now.day)
        stamp = '%02d%02d%02d' % time_tup
    elif format_ == 'printable':
        time_tup = (now.hour, now.minute, now.second, now.year, now.month, now.day)
        time_format = '%02d:%02d:%02d %02d/%02d/%02d'
        stamp = time_format % time_tup
    else:
        if use_second:
            time_tup = (now.year, now.month, now.day, now.hour, now.minute, now.second)
            time_formats = {
                'filename': 'ymd_hms-%04d-%02d-%02d_%02d-%02d-%02d',
                'comment': '# (yyyy-mm-dd hh:mm:ss) %04d-%02d-%02d %02d:%02d:%02d'}
        else:
            time_tup = (now.year, now.month, now.day, now.hour, now.minute)
            time_formats = {
                'filename': 'ymd_hm-%04d-%02d-%02d_%02d-%02d',
                'comment': '# (yyyy-mm-dd hh:mm) %04d-%02d-%02d %02d:%02d'}
        stamp = time_formats[format_] % time_tup
    if timezone:
        if isutc:
            stamp += '_UTC'
        else:
            from pytz import reference
            localtime = reference.LocalTimezone()
            tzname = localtime.tzname(now)
            stamp += '_' + tzname
    return stamp


def isect(list1, list2):
    """
    returns list1 elements that are also in list2. preserves order of list1

    intersect_ordered

    Args:
        list1 (list):
        list2 (list):

    Returns:
        list: new_list

    Example:
        >>> list1 = ['featweight_rowid', 'feature_rowid', 'config_rowid', 'featweight_forground_weight']
        >>> list2 = [u'featweight_rowid']
        >>> result = isect(list1, list2)
        >>> print(result)
        ['featweight_rowid']
    """
    set2 = set(list2)
    return [item for item in list1 if item in set2]


def safe_extreme(arr, op, fill=np.nan, finite=False, nans=True):
    """
    Applies an exterme operation to an 1d array (typically max/min) but ensures
    a value is always returned even in operations without identities. The
    default identity must be specified using the `fill` argument.

    Args:
        arr (ndarray): 1d array to take extreme of
        op (func): vectorized operation like np.max to apply to array
        fill (float): return type if arr has no elements (default = nan)
        finite (bool): if True ignores non-finite values (default = False)
        nans (bool): if False ignores nans (default = True)
    """
    if arr is None:
        extreme = fill
    else:
        arr = np.asarray(arr)
        if finite:
            arr = arr.compress(np.isfinite(arr))
        if not nans:
            arr = arr.compress(np.logical_not(np.isnan(arr)))
        if len(arr) == 0:
            extreme =  fill
        else:
            extreme = op(arr)
    return extreme


def safe_argmax(arr, fill=np.nan, finite=False, nans=True):
    """
    Doctest:
        >>> assert safe_argmax([np.nan, np.nan], nans=False) == 0
        >>> assert safe_argmax([-100, np.nan], nans=False) == 0
        >>> assert safe_argmax([np.nan, -100], nans=False) == 1
        >>> assert safe_argmax([-100, 0], nans=False) == 1
        >>> assert np.isnan(safe_argmax([]))
    """
    if len(arr) == 0:
        return fill
    extreme = safe_max(arr, fill=fill, finite=finite, nans=nans)
    if np.isnan(extreme):
        arg_extreme = np.where(np.isnan(arr))[0][0]
    else:
        arg_extreme = np.where(arr == extreme)[0][0]
    return arg_extreme


def safe_max(arr, fill=np.nan, finite=False, nans=True):
    r"""
    Args:
        arr (ndarray): 1d array to take max of
        fill (float): return type if arr has no elements (default = nan)
        finite (bool): if True ignores non-finite values (default = False)
        nans (bool): if False ignores nans (default = True)

    Example:
        >>> arrs = [[], [np.nan], [-np.inf, np.nan, np.inf], [np.inf], [np.inf, 1], [0, 1]]
        >>> arrs = [np.array(arr) for arr in arrs]
        >>> fill = np.nan
        >>> results1 = [safe_max(arr, fill, finite=False, nans=True) for arr in arrs]
        >>> results2 = [safe_max(arr, fill, finite=True, nans=True) for arr in arrs]
        >>> results3 = [safe_max(arr, fill, finite=True, nans=False) for arr in arrs]
        >>> results4 = [safe_max(arr, fill, finite=False, nans=False) for arr in arrs]
        >>> results = [results1, results2, results3, results4]
        >>> result = ('results = %s' % (ub.urepr(results, nl=1, sv=1),))
        >>> print(result)
        results = [
            [nan, nan, nan, inf, inf, 1],
            [nan, nan, nan, nan, 1.0, 1],
            [nan, nan, nan, nan, 1.0, 1],
            [nan, nan, inf, inf, inf, 1],
        ]
    """
    return safe_extreme(arr, np.max, fill, finite, nans)


def safe_min(arr, fill=np.nan, finite=False, nans=True):
    """
    Example:
        >>> arrs = [[], [np.nan], [-np.inf, np.nan, np.inf], [np.inf], [np.inf, 1], [0, 1]]
        >>> arrs = [np.array(arr) for arr in arrs]
        >>> fill = np.nan
        >>> results1 = [safe_min(arr, fill, finite=False, nans=True) for arr in arrs]
        >>> results2 = [safe_min(arr, fill, finite=True, nans=True) for arr in arrs]
        >>> results3 = [safe_min(arr, fill, finite=True, nans=False) for arr in arrs]
        >>> results4 = [safe_min(arr, fill, finite=False, nans=False) for arr in arrs]
        >>> results = [results1, results2, results3, results4]
        >>> result = ('results = %s' % (ub.urepr(results, nl=1, sv=1),))
        >>> print(result)
        results = [
            [nan, nan, nan, inf, 1.0, 0],
            [nan, nan, nan, nan, 1.0, 0],
            [nan, nan, nan, nan, 1.0, 0],
            [nan, nan, -inf, inf, 1.0, 0],
        ]
    """
    return safe_extreme(arr, np.min, fill, finite, nans)


def stats_dict(list_, axis=None, use_nan=False, use_sum=False,
               use_median=False, size=False):
    """
    Args:
        list_ (listlike): values to get statistics of
        axis (int): if `list_` is ndarray then this specifies the axis

    Returns:
        OrderedDict: stats: dictionary of common numpy statistics
            (min, max, mean, std, nMin, nMax, shape)

    Examples0:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> import numpy as np
        >>> axis = 0
        >>> np.random.seed(0)
        >>> list_ = np.random.rand(10, 2).astype(np.float32)
        >>> stats = stats_dict(list_, axis, use_nan=False)
        >>> result = str(ub.urepr(stats, nl=1, precision=4, with_dtype=True))
        >>> print(result)
        {
            'mean': np.array([0.5206, 0.6425], dtype=np.float32),
            'std': np.array([0.2854, 0.2517], dtype=np.float32),
            'max': np.array([0.9637, 0.9256], dtype=np.float32),
            'min': np.array([0.0202, 0.0871], dtype=np.float32),
            'nMin': np.array([1, 1], dtype=np.int32),
            'nMax': np.array([1, 1], dtype=np.int32),
            'shape': (10, 2),
        }

    Examples1:
        >>> import numpy as np
        >>> axis = 0
        >>> rng = np.random.RandomState(0)
        >>> list_ = rng.randint(0, 42, size=100).astype(np.float32)
        >>> list_[4] = np.nan
        >>> stats = stats_dict(list_, axis, use_nan=True)
        >>> result = str(ub.urepr(stats, precision=1, sk=True))
        >>> print(result)
        {mean: 20.0, std: 13.2, max: 41.0, min: 0.0, nMin: 7, nMax: 3, shape: (100,), num_nan: 1,}
    """
    datacast = np.float32
    # Assure input is in numpy format
    if isinstance(list_, np.ndarray):
        nparr = list_
    elif isinstance(list_, list):
        nparr = np.array(list_)
    else:
        nparr = np.array(list(list_))
    # Check to make sure stats are feasible
    if len(nparr) == 0:
        stats = ub.odict([('empty_list', True)])
        if size:
            stats['size'] = 0
    else:
        if use_nan:
            min_val = np.nanmin(nparr, axis=axis)
            max_val = np.nanmax(nparr, axis=axis)
            mean_ = np.nanmean(nparr, axis=axis)
            std_  = np.nanstd(nparr, axis=axis)
        else:
            min_val = nparr.min(axis=axis)
            max_val = nparr.max(axis=axis)
            mean_ = nparr.mean(axis=axis)
            std_  = nparr.std(axis=axis)
        # number of entries with min/max val
        nMin = np.sum(nparr == min_val, axis=axis)
        nMax = np.sum(nparr == max_val, axis=axis)
        stats = ub.odict([
            ('mean',  datacast(mean_)),
            ('std',   datacast(std_)),
            ('max',   (max_val)),
            ('min',   (min_val)),
            ('nMin',  np.int32(nMin)),
            ('nMax',  np.int32(nMax)),
        ])
        if size:
            stats['size'] = nparr.size
        else:
            stats['shape'] = nparr.shape
        if use_median:
            stats['med'] = np.nanmedian(nparr)
        if use_nan:
            stats['num_nan'] = np.isnan(nparr).sum()
        if use_sum:
            sumfunc = np.nansum if use_nan else np.sum
            stats['sum'] = sumfunc(nparr, axis=axis)
    return stats
