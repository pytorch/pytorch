import torch
import torch.fx
import inspect

def normalize_functional_args(gm : torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Normalize arguments to `torch.nn.functional` calls. This means
    that `args/kwargs` will be matched up to the functional's
    signature and rewritten into exclusively kwargs in positional
    order. Also populates default values. Does not support positional-
    only parameters or varargs parameters (*args, **kwargs).

    Args:
        gm (GraphModule): GraphModule to normalize

    Returns:

        GraphModule
    """
    for node in gm.graph.nodes:
        if node.op == 'call_function':
            if node.target.__module__ == 'torch.nn.functional':
                sig = inspect.signature(node.target)
                # Don't currently support positional-only
                # or varargs (*args, **kwargs) signatures
                supported_parameter_types = {
                    inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
                if any(p.kind not in supported_parameter_types for p in sig.parameters.values()):
                    continue
                bound_args = sig.bind(*node.args, **node.kwargs)
                bound_args.apply_defaults()

                new_kwargs : Dict[str, Any] = {}
                for param in sig.parameters:
                    new_kwargs[param] = bound_args.arguments[param]

                node.args = ()
                node.kwargs = new_kwargs

    gm.recompile()
    return gm
