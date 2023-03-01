from typing import Any, Dict, List, Optional

import torch.fx

__all__ = ["compile", "list_mode_optimizations", "list_optimizations"]


def compile(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    options: Optional[Dict[str, Any]] = None,
):
    """
    Compile a given FX graph with TorchInductor.  This allows compiling
    FX graphs captured without using TorchDynamo.

    Args:
        gm: The FX graph to compile.
        example_inputs:  List of tensor inputs.
        options:  Optional dict of config options.  See `torch._inductor.config`.

    Returns:
        Callable with same behavior as gm but faster.
    """
    from .compile_fx import compile_fx

    return compile_fx(gm, example_inputs, config_patches=options)


def list_mode_optimizations(mode: str = None) -> Dict[str, Any]:
    r"""Returns a dictionary describing the optimizations that each of the available
    modes passed to `torch.compile()` performs.

    Args:
        mode (str, optional): The mode to return the optimizations for.
        If None, returns optimizations for all modes

    Example::
        >>> torch.list_inductor_mode_optimizations()
    """

    mode_optimizations = {
        "default": {},
        "reduce-overhead": {
            "triton.cudagraphs": False,
            "size_asserts": False,
        },
        "max-autotune": {
            "epilogue_fusion": False,
            "max_autotune": False,
            "triton.cudagraphs": True,
        },
    }
    return mode_optimizations[mode] if mode else mode_optimizations


def list_optimizations() -> Dict[str, Any]:
    r"""Returns a dictionary describing the optimizations and debug configurations
    that are available to `torch.compile()`.

    The optimizations are documented in `torch._inductor.config`.

    Example::

        >>> torch.list_inductor_optimizations()
    """

    from torch._inductor import config

    current_config: Dict[str, Any] = config.to_dict()  # type: ignore[attr-defined]

    return list(current_config.keys())
