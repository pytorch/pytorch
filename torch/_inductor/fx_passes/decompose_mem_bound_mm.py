# mypy: allow-untyped-defs
import logging

import torch
from torch import Tensor
from torch._dynamo.utils import counters, is_node_meta_valid
from torch.fx.experimental.symbolic_shapes import (
    statically_known_false,
    statically_known_true,
)

from .. import config
from ..pattern_matcher import Arg, CallFunction, Match, register_graph_pattern
from ..utils import is_bf16x9_matmul
from .split_cat import construct_pattern_matcher_pass


aten = torch.ops.aten
log = logging.getLogger(__name__)

# TODO: need a better strategy for decomposing mm
# The following two constants are for CUDA device only
MIN_FIRST_DIMENSION_DECOMPOSITION = 10240
MAX_OTHER_DIMENSION_DECOMPOSITION = 32
# The following two constants are for CPU device only
CPU_MAX_FIRST_DIMENSION_DECOMPOSITION = 1
CPU_MAX_OTHER_DIMENSION_DECOMPOSITION = 2048

min_first_dimension_decomposition = MIN_FIRST_DIMENSION_DECOMPOSITION
max_other_dimension_decomposition = MAX_OTHER_DIMENSION_DECOMPOSITION
cpu_max_first_dimension_decomposition = CPU_MAX_FIRST_DIMENSION_DECOMPOSITION
cpu_max_other_dimension_decomposition = CPU_MAX_OTHER_DIMENSION_DECOMPOSITION
if "decompose_mm_pass" in config.post_grad_fusion_options:
    min_first_dimension_decomposition = config.post_grad_fusion_options[
        "decompose_mm_pass"
    ].get("min_first_dimension_decomposition", MIN_FIRST_DIMENSION_DECOMPOSITION)
    max_other_dimension_decomposition = config.post_grad_fusion_options[
        "decompose_mm_pass"
    ].get("max_other_dimension_decomposition", MAX_OTHER_DIMENSION_DECOMPOSITION)
    cpu_max_first_dimension_decomposition = config.post_grad_fusion_options[
        "decompose_mm_pass"
    ].get(
        "cpu_max_first_dimension_decomposition", CPU_MAX_FIRST_DIMENSION_DECOMPOSITION
    )
    cpu_max_other_dimension_decomposition = config.post_grad_fusion_options[
        "decompose_mm_pass"
    ].get(
        "cpu_max_other_dimension_decomposition", CPU_MAX_OTHER_DIMENSION_DECOMPOSITION
    )


# Decomposition policy of a device type: "accelerator" devices (e.g. "cuda",
# "xpu") decompose large-K matrix multiplications, while "cpu" devices
# decompose small matrices. Third-party backends can register their own
# device type via register_mm_decomposition_policy.
_ACCELERATOR_POLICY = "accelerator"
_CPU_POLICY = "cpu"

_mm_decomposition_policies: dict[str, str] = {}


def register_mm_decomposition_policy(device_type: str, policy: str) -> None:
    """Register `device_type` for the decompose_mm_pass post-grad pattern.

    Args:
        device_type: the device type string, e.g. "cuda", "cpu", or the name
            of a third-party backend registered via PrivateUse1.
        policy: either "accelerator" or "cpu", selecting the decomposition
            threshold logic to apply for that device.
    """
    _mm_decomposition_policies[device_type] = policy


register_mm_decomposition_policy("cuda", _ACCELERATOR_POLICY)
register_mm_decomposition_policy("xpu", _ACCELERATOR_POLICY)
register_mm_decomposition_policy("cpu", _CPU_POLICY)


def _get_decomposition_policy(a: Tensor, b: Tensor) -> str | None:
    """Return the decomposition policy for the device type shared by `a` and
    `b`, or None if they are on different device types or the device type is
    not registered."""
    if a.device.type != b.device.type:
        return None
    return _mm_decomposition_policies.get(a.device.type)


def check_device(a: Tensor, b: Tensor, device="cuda") -> bool:
    return (a.device.type == b.device.type) and (b.device.type == device)


def realize_inputs(inputs: list[torch.fx.Node]):
    for inp in inputs:
        if isinstance(inp, torch.fx.node.Node):
            inp.meta["inductor_realize_to_strides"] = True


def should_decompose_bmm(mat1, mat2) -> bool:
    if is_node_meta_valid(mat1) and is_node_meta_valid(mat2):
        mat1 = mat1.meta["val"]
        mat2 = mat2.meta["val"]
    else:
        return False
    if (
        len(mat1.shape) != 3
        or len(mat2.shape) != 3
        or is_bf16x9_matmul(mat1.device.type, mat1.dtype)
    ):
        return False
    policy = _get_decomposition_policy(mat1, mat2)
    if policy == _ACCELERATOR_POLICY:
        if mat1.shape[0] < min_first_dimension_decomposition:
            return False
        # 2 of m, n, k must be <= MAX_OTHER_DIMENSION_DECOMPOSITION
        # use bool() to deal with BooleanAtom type
        if (
            bool(mat1.shape[1] < max_other_dimension_decomposition)
            + bool(mat1.shape[2] < max_other_dimension_decomposition)
            + bool(mat2.shape[2] < max_other_dimension_decomposition)
            < 2
        ):
            return False
        return True
    elif policy == _CPU_POLICY:
        if config.post_grad_fusion_options["decompose_mm_pass"].get(
            "bmm_skip_dynamic_shape_dim_check", False
        ):
            return (
                statically_known_true(
                    mat1.shape[1] <= cpu_max_other_dimension_decomposition
                )
                and statically_known_true(
                    mat1.shape[2] <= cpu_max_other_dimension_decomposition
                )
                and statically_known_true(
                    mat2.shape[2] <= cpu_max_other_dimension_decomposition
                )
            )
        if (
            mat1.shape[0] <= cpu_max_first_dimension_decomposition
            and mat2.shape[0] <= cpu_max_first_dimension_decomposition
        ):
            return True
    return False


def should_decompose_mm(mat1, mat2) -> bool:
    """
    Determines whether matrix multiplication (mm) should be decomposed into pointwise operations
    based on the input matrices' metadata, shapes, device placement, and configuration options.
    Args:
        mat1: The first matrix operand. Expected to be an object with a `.meta` attribute containing
              a "val" key, or a tensor-like object with a `.shape` attribute.
        mat2: The second matrix operand. Same requirements as `mat1`.
    Returns:
        bool: True if the matrix multiplication should be decomposed according to the following logic:
            - Both inputs must have valid node metadata.
            - Both matrices must be 2-dimensional.
            - If the configuration option `skip_dynamic_shape_dim_check` is False:
                - Decomposition is only considered for statically-shaped matrices.
                - For CUDA devices: `mat1.shape[0]` must be at least `min_first_dimension_decomposition`,
                  and both dimensions of `mat2` must be less than `max_other_dimension_decomposition`.
                - For CPU devices: All relevant dimensions must be less than or equal to their respective
                  CPU decomposition thresholds.
            - If `skip_dynamic_shape_dim_check` is True:
                - Decomposition is considered for dynamic shapes as well, using a combination of
                  `statically_known_true` and `statically_known_false` checks to handle uncertainty.
                - The same dimension and device checks apply, but allow for dynamic/static uncertainty.
            - Returns False if any of the above conditions are not met.
    Notes:
        - Relies on helper functions such as `is_node_meta_valid`, `check_device`, `statically_known_true`,
          and `statically_known_false`, as well as configuration values like
          `min_first_dimension_decomposition`, `max_other_dimension_decomposition`, etc.
        - Designed for use in graph optimization or fusion passes where decomposing large or dynamic
          matrix multiplications can improve performance or memory usage.
    """
    if is_node_meta_valid(mat1) and is_node_meta_valid(mat2):
        mat1 = mat1.meta["val"]
        mat2 = mat2.meta["val"]
    else:
        return False
    if (
        len(mat1.shape) != 2
        or len(mat2.shape) != 2
        or is_bf16x9_matmul(mat1.device.type, mat1.dtype)
    ):
        return False
    # case 1: we skip decompose mm if the input is dynamic shape
    if not config.post_grad_fusion_options["decompose_mm_pass"].get(
        "skip_dynamic_shape_dim_check", False
    ):
        policy = _get_decomposition_policy(mat1, mat2)
        return (
            policy == _ACCELERATOR_POLICY
            and statically_known_true(
                mat1.shape[0] >= min_first_dimension_decomposition
            )
            and statically_known_true(mat2.shape[0] < max_other_dimension_decomposition)
            and statically_known_true(mat2.shape[1] < max_other_dimension_decomposition)
        ) or (
            policy == _CPU_POLICY
            and statically_known_true(
                mat1.shape[0] <= cpu_max_first_dimension_decomposition
            )
            and statically_known_true(
                mat2.shape[0] <= cpu_max_other_dimension_decomposition
            )
            and statically_known_true(
                mat2.shape[1] <= cpu_max_other_dimension_decomposition
            )
        )
    # case 2: we decompose mm if the input is dynamic shape
    else:
        policy = _get_decomposition_policy(mat1, mat2)
        return (
            policy == _ACCELERATOR_POLICY
            and (
                statically_known_true(
                    mat1.shape[0] >= min_first_dimension_decomposition
                )
                or not statically_known_false(
                    mat1.shape[0] >= min_first_dimension_decomposition
                )
            )
            and (
                statically_known_true(mat2.shape[0] < max_other_dimension_decomposition)
                or not statically_known_false(
                    mat2.shape[0] < max_other_dimension_decomposition
                )
            )
            and (
                statically_known_true(mat2.shape[1] < max_other_dimension_decomposition)
                or not statically_known_false(
                    mat2.shape[1] < max_other_dimension_decomposition
                )
            )
        ) or (
            policy == _CPU_POLICY
            and (
                statically_known_true(
                    mat1.shape[0] <= cpu_max_first_dimension_decomposition
                )
                or not statically_known_false(
                    mat1.shape[0] <= cpu_max_first_dimension_decomposition
                )
            )
            and (
                statically_known_true(
                    mat2.shape[0] <= cpu_max_other_dimension_decomposition
                )
                or not statically_known_false(
                    mat2.shape[0] <= cpu_max_other_dimension_decomposition
                )
            )
            and (
                statically_known_true(
                    mat2.shape[1] <= cpu_max_other_dimension_decomposition
                )
                or not statically_known_false(
                    mat2.shape[1] <= cpu_max_other_dimension_decomposition
                )
            )
        )


def print_decompose_pattern(match: Match, inputs: list[torch.fx.Node]):
    node = match.nodes[-1]
    log.debug(
        "Decompose %s with input shape: %s",
        node.target,
        ", ".join(
            str(input.meta["val"].shape) if "val" in input.meta else "None"
            for input in inputs
        ),
    )


@register_graph_pattern(
    CallFunction(aten.bmm, Arg(), Arg()),
    pass_dict=construct_pattern_matcher_pass("decompose_mm_pass"),
)
def decompose_bmm(match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node):
    def repl(mat1, mat2):
        return torch.sum(mat1[:, :, :, None] * mat2[:, None, :, :], dim=-2).to(
            mat1.dtype
        )

    if should_decompose_bmm(mat1, mat2):
        counters["inductor"]["decompose_bmm"] += 1
        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(repl, [mat1, mat2])
        print_decompose_pattern(match, [mat1, mat2])
        realize_inputs([mat1, mat2])
    return


@register_graph_pattern(
    CallFunction(aten.addmm, Arg(), Arg(), Arg()),
    pass_dict=construct_pattern_matcher_pass("decompose_mm_pass"),
)
def decompose_addmm(
    match: Match,
    mat1: torch.fx.Node,
    mat2: torch.fx.Node,
    mat3: torch.fx.Node,
):
    def repl(mat1, mat2, mat3):
        return (
            torch.sum(mat2[:, :, None] * mat3[None, :, :], dim=-2).to(mat2.dtype) + mat1
        )

    if should_decompose_mm(mat2, mat3):
        counters["inductor"]["decompose_addmm"] += 1
        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(repl, [mat1, mat2, mat3])
        print_decompose_pattern(match, [mat1, mat2, mat3])
        realize_inputs([mat1, mat2, mat3])
    return


@register_graph_pattern(
    CallFunction(aten.mm, Arg(), Arg()),
    pass_dict=construct_pattern_matcher_pass("decompose_mm_pass"),
)
def decompose_mm(
    match: Match,
    mat1: torch.fx.Node,
    mat2: torch.fx.Node,
):
    def repl(mat1, mat2):
        return torch.sum(mat1[:, :, None] * mat2[None, :, :], dim=-2).to(mat1.dtype)

    if should_decompose_mm(mat1, mat2):
        counters["inductor"]["decompose_mm"] += 1
        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(repl, [mat1, mat2])
        print_decompose_pattern(match, [mat1, mat2])
        realize_inputs([mat1, mat2])
    return
