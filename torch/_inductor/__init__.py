from typing import Any, Dict, List, Optional

import torch.fx

__all__ = ["compile", "list_mode_options", "list_options"]


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


def aot_compile(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Ahead-of-time compile a given FX graph with TorchInductor into a shared library.

    Args:
        gm: The FX graph to compile.
        example_inputs:  List of tensor inputs.
        options:  Optional dict of config options.  See `torch._inductor.config`.

    Returns:
        Path to the generated shared library
    """
    from .compile_fx import compile_fx_aot

    compiled = compile_fx_aot(
        gm,
        example_inputs,
        config_patches=options,
    )
    lib_path = compiled()
    return lib_path


def list_mode_options(mode: str = None) -> Dict[str, Any]:
    r"""Returns a dictionary describing the optimizations that each of the available
    modes passed to `torch.compile()` performs.

    Args:
        mode (str, optional): The mode to return the optimizations for.
        If None, returns optimizations for all modes

    Example::
        >>> torch._inductor.list_mode_options()
    """

    mode_options = {
        "default": {},
        "reduce-overhead": {
            "triton.cudagraphs": True,
        },
        "max-autotune": {
            "epilogue_fusion": True,
            "max_autotune": True,
            "triton.cudagraphs": True,
        },
    }
    return mode_options[mode] if mode else mode_options


def list_options() -> Dict[str, Any]:
    r"""Returns a dictionary describing the optimizations and debug configurations
    that are available to `torch.compile()`.

    The options are documented in `torch._inductor.config`.

    Example::

        >>> torch._inductor.list_options()
    """

    from torch._inductor import config

    current_config: Dict[str, Any] = config.to_dict()  # type: ignore[attr-defined]

    return list(current_config.keys())
