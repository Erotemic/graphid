class KWSpec:
    """
    Safer keyword arguments with keyword specifications.
    """
    def __init__(kwspec, spec):
        kwspec.spec = spec
        kwspec.spec_set = set(spec)

    def __call__(kwspec, **kwargs):
        kw = kwspec.spec.copy()
        unknown_keys = set(kw).difference(kwspec.spec_set)
        if unknown_keys:
            raise KeyError('Unknown keys: {}'.format(unknown_keys))
        kw.update(kwargs)
        return kw
