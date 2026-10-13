# mypy: allow-untyped-defs

import logging
from typing import TYPE_CHECKING

import torch

from .. import config as inductor_config
from ..kernel_inputs import MMKernelInputs
from ..lowering import lowerings
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    TritonTemplate,
)
from ..utils import use_aten_gemm_kernels, use_triton_template
from ..virtualized import V
from .mm_common import load_kernel_template, mm_args, mm_grid


if TYPE_CHECKING:
    from torch._inductor.ir import ChoiceCaller
    from torch._inductor.select_algorithm import KernelTemplate

log = logging.getLogger(__name__)

aten = torch.ops.aten

aten_mm_plus_mm = ExternKernelChoice(
    torch.ops.inductor._mm_plus_mm, "torch::inductor::_mm_plus_mm"
)

# The mm_plus_mm Triton template. XPU renders the K loops ascending via the
# ASCENDING_K kwarg (see MMTemplateConfigMixin.ascending_k).
mm_plus_mm_template = TritonTemplate(
    name="mm_plus_mm",
    grid=mm_grid,
    debug=False,
    source=load_kernel_template("triton_mm_plus_mm"),
    cache_codegen_enabled_for_template=True,
)


def tuned_mm_plus_mm(mat1, mat2, mat3, mat4, *, layout=None):
    """
    Computes mm(mat1, mat2) + mm(mat3, mat4)
    """
    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m1, n1, k1, layout1, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    m2, n2, _, layout2, mat3, mat4 = mm_args(mat3, mat4, layout=layout)

    # Optimization is optional, because we can always just not do the fusion
    if (
        m1 * n1 == 0
        or m2 * n2 == 0
        or not V.graph.sizevars.statically_known_list_equals(
            mat1.get_size(), mat3.get_size()
        )
        or not V.graph.sizevars.statically_known_list_equals(
            mat2.get_size(), mat4.get_size()
        )
        or inductor_config.triton.native_matmul
    ):
        # TODO(jansel): support different K values when this is fixed:
        # https://github.com/triton-lang/triton/issues/967
        return lowerings[aten.add](
            lowerings[aten.mm](mat1, mat2), lowerings[aten.mm](mat3, mat4)
        )

    # Create MMKernelInputs for MM Plus MM (matrices are at indices 0, 1 for first pair)
    # Note: This is a special case with 4 matrices, but we use the first pair for M, N, K extraction
    kernel_inputs = MMKernelInputs([mat1, mat2, mat3, mat4], mat1_idx=0, mat2_idx=1)

    if layout1 != layout2:
        raise AssertionError("layout1 and layout2 must be equal")
    # options to tune from
    choices: list[ChoiceCaller] = []

    # Collect all templates for unified call
    templates_to_use: list[ExternKernelChoice | KernelTemplate] = []
    if use_aten_gemm_kernels():
        templates_to_use.append(aten_mm_plus_mm)

    if use_triton_template(layout1, check_max_autotune=False):
        templates_to_use.append(mm_plus_mm_template)

    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(kernel_inputs, templates_to_use, "mm_plus_mm")
    )

    node, _ = autotune_select_algorithm(
        "mm_plus_mm", choices, kernel_inputs.nodes(), layout1
    )
    return node
