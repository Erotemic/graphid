import ubelt as ub
import numpy as np
import itertools as it
from graphid.core.state import POSTV, NEGTV, INCMP, UNREV, UNKWN  # NOQA
from graphid import util


def demodata_mtest_infr(state='empty'):
    import ibeis
    from graphid.core.annot_inference import AnnotInference
    ibs = ibeis.opendb(db='PZ_MTEST')
    annots = ibs.annots()
    names = list(annots.group_items(annots.nids).values())
    util.shuffle(names, rng=321)
    test_aids = list(ub.flatten(names[1::2]))
    infr = AnnotInference(ibs, test_aids, autoinit=True)
    infr.reset(state=state)
    return infr


def demodata_infr2(defaultdb='PZ_MTEST'):
    import ibeis
    from graphid.core.annot_inference import AnnotInference
    defaultdb = 'PZ_MTEST'
    ibs = ibeis.opendb(defaultdb=defaultdb)
    annots = ibs.annots()
    names = list(annots.group_items(annots.nids).values())[0:20]
    def dummy_phi(c, n):
        x = np.arange(n)
        phi = c * x / (c * x + 1)
        phi = phi / phi.sum()
        phi = np.diff(phi)
        return phi
    phis = {
        c: dummy_phi(c, 30)
        for c in range(1, 4)
    }
    aids = list(ub.flatten(names))
    infr = AnnotInference(ibs, aids, autoinit=True)
    infr.init_termination_criteria(phis)
    infr.init_refresh_criteria()

    # Partially review
    n1, n2, n3, n4 = names[0:4]
    for name in names[4:]:
        for a, b in ub.iter_window(name.aids, 2):
            infr.add_feedback((a, b), POSTV)

    for name1, name2 in it.combinations(names[4:], 2):
        infr.add_feedback((name1.aids[0], name2.aids[0]), NEGTV)
    return infr
