# -*- coding: utf-8 -*-
"""
Regenerate Input Command
mkinit graphid.demo --lazy_loader_typed
"""
import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

__all__ = ['DummyRanker', 'DummyVerif', 'demo_script', 'demodata_infr',
           'dummy_algos', 'dummy_infr', 'run_demo']
