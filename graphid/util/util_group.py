import ubelt as ub
import operator as op


def sortedby(item_list, key_list, reverse=False):
    """ sorts ``item_list`` using key_list

    Args:
        list_ (list): list to sort
        key_list (list): list to sort by
        reverse (bool): sort order is descending (largest first)
                        if reverse is True else acscending (smallest first)

    Returns:
        list : ``list_`` sorted by the values of another ``list``. defaults to
        ascending order

    SeeAlso:
        sortedby2

    Examples:
        >>> list_    = [1, 2, 3, 4, 5]
        >>> key_list = [2, 5, 3, 1, 5]
        >>> result = sortedby(list_, key_list, reverse=True)
        >>> print(result)
        [5, 2, 3, 1, 4]

    """
    assert len(item_list) == len(key_list), (
        'Expected same len. Got: %r != %r' % (len(item_list), len(key_list)))
    sorted_list = [item for (key, item) in
                   sorted(list(zip(key_list, item_list)), reverse=reverse)]
    return sorted_list


def grouping_delta(old, new, pure=True):
    """
    Finds what happened to the old groups to form the new groups.

    Args:
        old (set of frozensets): old grouping
        new (set of frozensets): new grouping
        pure (bool): hybrids are separated from pure merges and splits if
            pure is True, otherwise hybrid cases are grouped in merges and
            splits.

    Returns:
        dict: delta: dictionary of changes containing the merges, splits,
            unchanged, and hybrid cases. Except for unchanged, case a subdict
            with new and old keys.  For splits / merges, one of these contains
            nested sequences to indicate what the split / merge is. Also
            reports elements added and removed between old and new if the
            flattened sets are not the same.

    Notes:
        merges - which old groups were merged into a single new group.
        splits - which old groups were split into multiple new groups.
        hybrid - which old groups had split/merge actions applied.
        unchanged - which old groups are the same as new groups.

    Example:
        >>> # xdoc: +IGNORE_WHITESPACE
        >>> old = [
        >>>     [20, 21, 22, 23], [1, 2], [12], [13, 14], [3, 4], [5, 6,11],
        >>>     [7], [8, 9], [10], [31, 32], [33, 34, 35], [41, 42, 43, 44, 45]
        >>> ]
        >>> new = [
        >>>   [20, 21], [22, 23], [1, 2], [12, 13, 14], [4], [5, 6, 3], [7, 8],
        >>>   [9, 10, 11], [31, 32, 33, 34, 35],   [41, 42, 43, 44], [45],
        >>> ]
        >>> delta = grouping_delta(old, new)
        >>> assert set(old[0]) in delta['splits']['old']
        >>> assert set(new[3]) in delta['merges']['new']
        >>> assert set(old[1]) in delta['unchanged']
        >>> result = ub.urepr(delta, nl=2, sort=True, nobr=1, sk=True)
        >>> print(result)
        hybrid: {
            merges: [{{10}, {11}, {9}}, {{3}, {5, 6}}, {{4}}, {{7}, {8}}],
            new: {{3, 5, 6}, {4}, {7, 8}, {9, 10, 11}},
            old: {{10}, {3, 4}, {5, 6, 11}, {7}, {8, 9}},
            splits: [{{10}}, {{11}, {5, 6}}, {{3}, {4}}, {{7}}, {{8}, {9}}],
        },
        items: {
            added: {},
            removed: {},
        },
        merges: {
            new: [{12, 13, 14}, {31, 32, 33, 34, 35}],
            old: [{{12}, {13, 14}}, {{31, 32}, {33, 34, 35}}],
        },
        splits: {
            new: [{{20, 21}, {22, 23}}, {{41, 42, 43, 44}, {45}}],
            old: [{20, 21, 22, 23}, {41, 42, 43, 44, 45}],
        },
        unchanged: {
            {1, 2},
        },


    Example:
        >>> old = [
        >>>     [1, 2, 3], [4], [5, 6, 7, 8, 9], [10, 11, 12]
        >>> ]
        >>> new = [
        >>>     [1], [2], [3, 4], [5, 6, 7], [8, 9, 10, 11, 12]
        >>> ]
        >>> # every case here is hybrid
        >>> pure_delta = grouping_delta(old, new, pure=True)
        >>> assert len(list(ub.flatten(pure_delta['merges'].values()))) == 0
        >>> assert len(list(ub.flatten(pure_delta['splits'].values()))) == 0
        >>> delta = grouping_delta(old, new, pure=False)
        >>> delta = order_dict_by(delta, ['unchanged', 'splits', 'merges'])
        >>> result = ub.urepr(delta, nl=2, sort=True, sk=True)
        >>> print(result)
        {
            items: {
                added: {},
                removed: {},
            },
            merges: [
                [{3}, {4}],
                [{10, 11, 12}, {8, 9}],
            ],
            splits: [
                [{1}, {2}, {3}],
                [{5, 6, 7}, {8, 9}],
            ],
            unchanged: {},
        }


    Example:
        >>> delta = grouping_delta([[1, 2, 3]], [])
        >>> assert len(delta['items']['removed']) == 3
        >>> delta = grouping_delta([], [[1, 2, 3]])
        >>> assert len(delta['items']['added']) == 3
        >>> delta = grouping_delta([[1]], [[1, 2, 3]])
        >>> assert len(delta['items']['added']) == 2
        >>> assert len(delta['unchanged']) == 1
    """
    _old = {frozenset(_group) for _group in old}
    _new = {frozenset(_group) for _group in new}

    _new_items = set(ub.flatten(_new))
    _old_items = set(ub.flatten(_old))

    if _new_items != _old_items:
        _added = _new_items - _old_items
        _removed = _old_items - _new_items
        # Make the sets items the same
        _old = {frozenset(_group - _removed) for _group in _old}
        _new = {frozenset(_group - _added) for _group in _new}
        _old = {_group for _group in _old if _group}
        _new = {_group for _group in _new if _group}
    else:
        _added = {}
        _removed = {}

    # assert _new_items == _old_items, 'new and old sets must be the same'

    # Find the groups that are exactly the same
    unchanged = _new.intersection(_old)

    new_sets = _new.difference(unchanged)
    old_sets = _old.difference(unchanged)

    # connected compoment lookups
    old_conn = {p: frozenset(ps) for ps in _old for p in ps}
    new_conn = {t: frozenset(ts) for ts in _new for t in ts}

    # How many old sets can be merged into perfect pieces?
    # For each new sets, find if it can be made via merging old sets
    old_merges = []
    new_merges = []
    for ts in new_sets:
        ccs = set([old_conn.get(t, frozenset()) for t in ts])
        if frozenset.union(*ccs) == ts:
            # This is a pure merge
            old_merges.append(ccs)
            new_merges.append(ts)

    # How many oldictions can be split into perfect pieces?
    new_splits = []
    old_splits = []
    for ps in old_sets:
        ccs = set([new_conn.get(p, frozenset()) for p in ps])
        if frozenset.union(*ccs) == ps:
            # This is a pure merge
            new_splits.append(ccs)
            old_splits.append(ps)

    old_merges_flat = list(ub.flatten(old_merges))
    new_splits_flat = list(ub.flatten(new_splits))

    old_hybrid = frozenset(map(frozenset, old_sets)).difference(
        set(old_splits + old_merges_flat))

    new_hybrid = frozenset(map(frozenset, new_sets)).difference(
        set(new_merges + new_splits_flat))

    breakup_hybrids = True
    if breakup_hybrids:
        # First split each hybrid
        lookup = {a: n for n, items in enumerate(new_hybrid) for a in items}
        hybrid_splits = []
        for items in old_hybrid:
            nids = list(ub.take(lookup, items))
            split_part = list(ub.group_items(items, nids).values())
            hybrid_splits.append(set(map(frozenset, split_part)))

        # And then merge them into new groups
        hybrid_merge_parts = list(ub.flatten(hybrid_splits))
        part_nids = [lookup[next(iter(aids))] for aids in hybrid_merge_parts]
        hybrid_merges = list(map(set, ub.group_items(hybrid_merge_parts,
                                                     part_nids).values()))

    if pure:
        delta = ub.odict()
        delta['unchanged'] = unchanged
        delta['splits'] = ub.odict([
            ('old', old_splits),
            ('new', new_splits),
        ])
        delta['merges'] = ub.odict([
            ('old', old_merges),
            ('new', new_merges),
        ])
        delta['hybrid'] = ub.odict([
            ('old', old_hybrid),
            ('new', new_hybrid),
            ('splits', hybrid_splits),
            ('merges', hybrid_merges),
        ])
    else:
        # Incorporate hybrid partial cases with pure splits and merges
        new_splits2 = [s for s in hybrid_splits if len(s) > 1]
        old_merges2 = [m for m in hybrid_merges if len(m) > 1]
        all_new_splits = new_splits + new_splits2
        all_old_merges = old_merges + old_merges2

        # Don't bother differentiating old and new
        # old_splits2 = [frozenset(ub.flatten(s)) for s in new_splits2]
        # new_merges2 = [frozenset(ub.flatten(m)) for m in old_merges2]
        # all_old_splits = old_splits + old_splits2
        # all_new_merges = new_merges + new_merges2

        splits = all_new_splits
        merges = all_old_merges

        # Sort by split and merge sizes
        splits = sortedby(splits, [len(list(ub.flatten(_))) for _ in splits])
        merges = sortedby(merges, [len(list(ub.flatten(_))) for _ in merges])
        splits = [sortedby(_, list(map(len, _))) for _ in splits]
        merges = [sortedby(_, list(map(len, _))) for _ in merges]

        delta = ub.odict()
        delta['unchanged'] = unchanged
        delta['splits'] = splits
        delta['merges'] = merges

    delta['items'] = ub.odict([
        ('added', _added),
        ('removed', _removed),
    ])
    return delta


def order_dict_by(dict_, key_order):
    r"""
    Reorders items in a dictionary according to a custom key order

    Args:
        dict_ (dict_):  a dictionary
        key_order (list): custom key order

    Returns:
        OrderedDict: sorted_dict

    Example:
        >>> dict_ = {1: 1, 2: 2, 3: 3, 4: 4}
        >>> key_order = [4, 2, 3, 1]
        >>> sorted_dict = order_dict_by(dict_, key_order)
        >>> result = ('sorted_dict = %s' % (ub.urepr(sorted_dict, nl=False),))
        >>> print(result)
        >>> assert result == 'sorted_dict = {4: 4, 2: 2, 3: 3, 1: 1}'
    """
    import itertools as it
    dict_keys = set(dict_.keys())
    other_keys = dict_keys - set(key_order)
    key_order = it.chain(key_order, other_keys)
    sorted_dict = ub.odict(
        (key, dict_[key]) for key in key_order if key in dict_keys
    )
    return sorted_dict


def group_pairs(pair_list):
    """
    Groups a list of items using the first element in each pair as the item and
    the second element as the groupid.

    Args:
        pair_list (list): list of 2-tuples (item, groupid)

    Returns:
        dict: groupid_to_items: maps a groupid to a list of items
    """
    # Initialize dict of lists
    groupid_to_items = ub.ddict(list)
    # Insert each item into the correct group
    for item, groupid in pair_list:
        groupid_to_items[groupid].append(item)
    return groupid_to_items


def sort_dict(dict_, part='keys', key=None, reverse=False):
    """
    sorts a dictionary by its values or its keys

    Args:
        dict_ (dict_):  a dictionary
        part (str): specifies to sort by keys or values
        key (Optional[func]): a function that takes specified part
            and returns a sortable value
        reverse (bool): (Defaults to False) - True for descinding order. False
            for ascending order.

    Returns:
        OrderedDict: sorted dictionary

    Example:
        >>> dict_ = {'a': 3, 'c': 2, 'b': 1}
        >>> results = []
        >>> results.append(sort_dict(dict_, 'keys'))
        >>> results.append(sort_dict(dict_, 'vals'))
        >>> results.append(sort_dict(dict_, 'vals', lambda x: -x))
        >>> result = ub.urepr(results)
        >>> print(result)
        [
            {'a': 3, 'b': 1, 'c': 2},
            {'b': 1, 'c': 2, 'a': 3},
            {'a': 3, 'c': 2, 'b': 1},
        ]
    """
    if part == 'keys':
        index = 0
    elif part in {'vals', 'values'}:
        index = 1
    else:
        raise ValueError('Unknown method part=%r' % (part,))
    if key is None:
        _key = op.itemgetter(index)
    else:
        def _key(item):
            return key(item[index])
    sorted_items = sorted(dict_.items(), key=_key, reverse=reverse)
    sorted_dict = ub.odict(sorted_items)
    return sorted_dict
