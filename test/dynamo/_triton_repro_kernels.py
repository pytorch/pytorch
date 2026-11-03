# Owner(s): ["module: dynamo"]

# User-defined Triton kernels at module scope whose constexpr globals and
# autotune arguments after_aot repros must reproduce.
import enum

import triton
import triton.language as tl


class Mode(enum.IntEnum):
    FAST = 1


IS_ENABLED = tl.constexpr(False)
OFFSET = tl.constexpr(0)
MODE = tl.constexpr(Mode.FAST)
SHAPE = tl.constexpr((0, 1))


@triton.jit
def kernel_with_constexpr_globals(x_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK) + OFFSET + SHAPE[0]
    values = tl.load(x_ptr + offsets)
    if IS_ENABLED or MODE != 1:
        values = values + 1
    tl.store(x_ptr + offsets, values)


def _prune_configs(configs, named_args, **kwargs):
    return configs


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2, maxnreg=128),
        triton.Config({"BLOCK": 128}, num_warps=8, num_stages=3, num_ctas=2),
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],
    restore_value=["x_ptr"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def autotuned_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.atomic_add(out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask), mask=mask)
