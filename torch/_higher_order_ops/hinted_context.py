import logging

import torch
from torch._ops import HigherOrderOperator
import torch._dynamo.config

log = logging.getLogger(__name__)


class HintedContext(HigherOrderOperator):
    def __init__(self):
        super().__init__("hinted_context")

    def __call__(self, gmod, *args, **kwargs):
        import torch.fx.traceback as fx_traceback
        from torch._higher_order_ops.hint_tracker import ContextHintTracker
        from torch.fx import Interpreter

        hint: str = ""
        if hasattr(gmod, "meta"):
            assert "hint" in gmod.meta
            hint = gmod.meta["hint"]

        # Put context over interpreter so we can gather all levels of context hints.
        with fx_traceback.preserve_node_meta(), ContextHintTracker(hint):
            return Interpreter(gmod).run(*args)


hinted_context = HintedContext()
