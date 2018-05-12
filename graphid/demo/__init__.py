# -*- coding: utf-8 -*-
"""
Regenerate Input Command
mkinit ~/code/graphid/graphid/demo
"""
# flake8: noqa
__DYNAMIC__ = True
if __DYNAMIC__:
    from mkinit import dynamic_mkinit
    exec(dynamic_mkinit.dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    from graphid.demo import demo_script
    from graphid.demo import dummy_algos
    from graphid.demo import dummy_infr
    from graphid.demo.demo_script import (run_demo,)
    from graphid.demo.dummy_algos import (DummyVerif,)
    from graphid.demo.dummy_infr import (demodata_infr,)
    # </AUTOGEN_INIT>
