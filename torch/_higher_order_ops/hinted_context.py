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
        from torch.fx import GraphModule
        from torch._higher_order_ops.hint_tracker import ContextHintTracker

        # Put context over interpreter so we can gather all levels of context hints.
        # This part can be called both as a GraphModule and a function. In case of
        # GraphModule, it will be created by HintedContextHOOVariable which puts meta
        # from kwargs there (in this case the hint is not available here in the kwargs).
        # In case of function, we need to take the hint directly from kwargs here.
        hint: str = ""
        if isinstance(gmod, GraphModule):
            from torch.fx import Interpreter

            if hasattr(gmod, "meta"):
                assert "hint" in gmod.meta
                hint = gmod.meta["hint"]

            with fx_traceback.preserve_node_meta(), ContextHintTracker(hint):
                return Interpreter(gmod).run(*args)
        else:
            # This should happen only when AOT traces through autograd overridden nodes.
            # [TODO] Can we double check this somehow?
            import torch._dynamo  # noqa: F401
            from torch._dynamo import disable

            @disable
            def wrapper():
                result = gmod(*args, **kwargs)
                return result

            if "hint" in kwargs:
                hint = kwargs["hint"]
            with fx_traceback.preserve_node_meta(), ContextHintTracker(hint):
                return wrapper()

hinted_context = HintedContext()
