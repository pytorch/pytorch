from typing import Tuple, Dict, Any
from torch.fx.experimental.normalize import NormalizeArgs
from torch.fx.node import Argument

class NormalizeArgsPreservingFQNs(NormalizeArgs):
    def call_module(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        module_qualified_name = target
        return self.tracer.create_proxy("call_module", module_qualified_name, args, kwargs)
