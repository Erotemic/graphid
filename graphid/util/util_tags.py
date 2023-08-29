import re
import operator
import numpy as np
import ubelt as ub


def tag_hist(tags_list):
    return ub.dict_hist(ub.flatten(tags_list), ordered=True)


def build_alias_map(regex_map, tag_vocab):
    """
    Constructs explicit mapping. Order of items in regex map matters.
    Items at top are given preference.
    """
    from graphid import util
    alias_map = ub.odict([])
    for pats, new_tag in reversed(regex_map):
        pats = util.ensure_iterable(pats)
        for pat in pats:
            flags = [re.match(pat, t) for t in tag_vocab]
            for old_tag in ub.compress(tag_vocab, flags):
                alias_map[old_tag] = new_tag
    identity_map = util.take_column(regex_map, 1)
    for tag in util.filter_Nones(identity_map):
        alias_map[tag] = tag
    return alias_map


def alias_tags(tags_list, alias_map):
    """
    update tags to new values

    Args:
        tags_list (list):
        alias_map (list): list of 2-tuples with regex, value

    Returns:
        list: updated tags
    """
    def _alias_dict(tags):
        tags_ = [alias_map.get(t, t) for t in tags]
        return list(set([t for t in tags_ if t is not None]))
    tags_list_ = [_alias_dict(tags) for tags in tags_list]
    return tags_list_


def filterflags_general_tags(tags_list, has_any=None, has_all=None,
                             has_none=None, min_num=None, max_num=None,
                             any_startswith=None, any_endswith=None,
                             in_any=None, any_match=None, none_match=None,
                             logic='and', ignore_case=True):
    r"""
    Args:
        tags_list (list):
        has_any (None): (default = None)
        has_all (None): (default = None)
        min_num (None): (default = None)
        max_num (None): (default = None)

    Notes:
        in_any should probably be ni_any

    Example1:
        >>> # ENABLE_DOCTEST
        >>> tags_list = [['v'], [], ['P'], ['P'], ['n', 'o'], [], ['n', 'N'], ['e', 'i', 'p', 'b', 'n'], ['n'], ['n'], ['N']]
        >>> has_all = 'n'
        >>> min_num = 1
        >>> flags = filterflags_general_tags(tags_list, has_all=has_all, min_num=min_num)
        >>> result = list(ub.compress(tags_list, flags))
        >>> print('result = %r' % (result,))

    Example2:
        >>> tags_list = [['vn'], ['vn', 'no'], ['P'], ['P'], ['n', 'o'], [], ['n', 'N'], ['e', 'i', 'p', 'b', 'n'], ['n'], ['n', 'nP'], ['NP']]
        >>> kwargs = {
        >>>     'any_endswith': 'n',
        >>>     'any_match': None,
        >>>     'any_startswith': 'n',
        >>>     'has_all': None,
        >>>     'has_any': None,
        >>>     'has_none': None,
        >>>     'max_num': 3,
        >>>     'min_num': 1,
        >>>     'none_match': ['P'],
        >>> }
        >>> flags = filterflags_general_tags(tags_list, **kwargs)
        >>> filtered = list(ub.compress(tags_list, flags))
        >>> result = ('result = %s' % (ub.urepr(filtered, nl=0),))
        >>> print(result)
        result = [['vn', 'no'], ['n', 'o'], ['n', 'N'], ['n'], ['n', 'nP']]
    """

    def _fix_tags(tags):
        if ignore_case:
            return set([]) if tags is None else {str(t.lower()) for t in tags}
        else:
            return set([]) if tags is None else {str() for t in tags}

    if logic is None:
        logic = 'and'

    logic_func = {
        'and': np.logical_and,
        'or': np.logical_or,
    }[logic]

    default_func = {
        'and': np.ones,
        'or': np.zeros,
    }[logic]

    tags_list_ = [_fix_tags(tags_) for tags_ in tags_list]
    flags = default_func(len(tags_list_), dtype=bool)

    if min_num is not None:
        flags_ = [len(tags_) >= min_num for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    if max_num is not None:
        flags_ = [len(tags_) <= max_num for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    if has_any is not None:
        from grpahid import util
        has_any = _fix_tags(set(util.ensure_iterable(has_any)))
        flags_ = [len(has_any.intersection(tags_)) > 0 for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    if has_none is not None:
        from grpahid import util
        has_none = _fix_tags(set(util.ensure_iterable(has_none)))
        flags_ = [len(has_none.intersection(tags_)) == 0 for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    if has_all is not None:
        from graphid import util
        has_all = _fix_tags(set(util.ensure_iterable(has_all)))
        flags_ = [len(has_all.intersection(tags_)) == len(has_all) for tags_ in tags_list_]
        logic_func(flags, flags_, out=flags)

    def _test_item(tags_, fields, op, compare):
        t_flags = [any([compare(t, f) for f in fields]) for t in tags_]
        num_passed = sum(t_flags)
        flag = op(num_passed, 0)
        return flag

    def _flag_tags(tags_list, fields, op, compare):
        flags = [_test_item(tags_, fields, op, compare) for tags_ in tags_list_]
        return flags

    def _exec_filter(flags, tags_list, fields, op, compare):
        if fields is not None:
            from graphid import util
            fields = util.ensure_iterable(fields)
            if ignore_case:
                fields = [f.lower() for f in fields]
            flags_ = _flag_tags(tags_list, fields, op, compare)
            logic_func(flags, flags_, out=flags)
        return flags

    flags = _exec_filter(
        flags, tags_list, any_startswith,
        operator.gt, str.startswith)

    flags = _exec_filter(
        flags, tags_list, in_any,
        operator.gt, operator.contains)

    flags = _exec_filter(
        flags, tags_list, any_endswith,
        operator.gt, str.endswith)

    flags = _exec_filter(
        flags, tags_list, any_match,
        operator.gt, lambda t, f: re.match(f, t))

    flags = _exec_filter(
        flags, tags_list, none_match,
        operator.eq, lambda t, f: re.match(f, t))
    return flags
