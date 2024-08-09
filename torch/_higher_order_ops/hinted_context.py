import logging
from typing import Any

from torch._ops import HigherOrderOperator
from torch.fx.graph_module import GraphModule

log = logging.getLogger(__name__)


class HintedContext(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("hinted_context")

    def __call__(self, gmod: GraphModule, *args: Any, **kwargs: Any) -> Any:
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
