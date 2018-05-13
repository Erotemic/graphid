# -*- coding: utf-8 -*-
"""
Regenerate Input Command
mkinit ~/code/graphid/graphid/core
"""
# flake8: noqa
__DYNAMIC__ = True
if __DYNAMIC__:
    from mkinit import dynamic_mkinit
    exec(dynamic_mkinit.dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    from graphid.core import annot_inference
    from graphid.core import mixin_callbacks
    from graphid.core import mixin_dynamic
    from graphid.core import mixin_helpers
    from graphid.core import mixin_loops
    from graphid.core import mixin_priority
    from graphid.core import mixin_redundancy
    from graphid.core import mixin_simulation
    from graphid.core import mixin_viz
    from graphid.core import refresh
    from graphid.core import state
    from graphid.core.annot_inference import (AltConstructors, AnnotInference,
                                              Consistency, Feedback, MiscHelpers,
                                              NameRelabel,)
    from graphid.core.mixin_callbacks import (InfrCallbacks, InfrCandidates,)
    from graphid.core.mixin_dynamic import (DynamicUpdate, NonDynamicUpdate,
                                            Recovery,)
    from graphid.core.mixin_helpers import (AssertInvariants, AttrAccess,
                                            Convenience, DEBUG_INCON, DummyEdges,)
    from graphid.core.mixin_loops import (InfrLoops, InfrReviewers,)
    from graphid.core.mixin_priority import (Priority,)
    from graphid.core.mixin_redundancy import (Redundancy,)
    from graphid.core.mixin_simulation import (SimulationHelpers, UserOracle,)
    from graphid.core.mixin_viz import (GraphVisualization, color_nodes,
                                        nx_ensure_agraph_color, on_pick,)
    from graphid.core.refresh import (RefreshCriteria, demo_refresh,)
    from graphid.core.state import (CONFIDENCE, DIFF, EVIDENCE_DECISION, INCMP,
                                    META_DECISION, NEGTV, NULL, POSTV, QUAL, SAME,
                                    UNINFERABLE, UNKWN, UNREV, VIEW,)
    # </AUTOGEN_INIT>
