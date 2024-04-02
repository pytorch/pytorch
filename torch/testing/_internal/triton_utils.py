import functools
import unittest

from torch.testing._internal.inductor_utils import HAS_CUDA

requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires cuda")

if HAS_CUDA:
    import triton
    from triton import language as tl

    # Define here so that multiple tests can take advantage of it
    @triton.jit
    def add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def add_kernel_with_optional_param(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        ARGS_PASSED: "tl.constexpr",
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        if ARGS_PASSED == "two":
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
        else:
            output = x
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 64}, num_stages=3, num_warps=8),
        ],
        key=[],
    )
    @triton.jit
    def add_kernel_autotuned(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_SIZE_X": 128, "BLOCK_SIZE_Y": 128}, num_stages=3, num_warps=8
            ),
            triton.Config(
                {"BLOCK_SIZE_X": 64, "BLOCK_SIZE_Y": 64}, num_stages=3, num_warps=8
            ),
        ],
        key=[],
    )
    @triton.jit
    def add_kernel_2d_autotuned(
        in_ptr0,
        in_ptr1,
        out_ptr,
        x_elements,
        y_elements,
        BLOCK_SIZE_X: "tl.constexpr",
        BLOCK_SIZE_Y: "tl.constexpr",
    ):
        xoffset = tl.program_id(0) * BLOCK_SIZE_X
        xindex = xoffset + tl.arange(0, BLOCK_SIZE_X)[:, None]
        xmask = xindex < x_elements
        yoffset = tl.program_id(1) * BLOCK_SIZE_Y
        yindex = yoffset + tl.arange(0, BLOCK_SIZE_Y)[None, :]
        ymask = yindex < y_elements
        x1 = xindex
        y0 = yindex
        tmp0 = tl.load(in_ptr0 + (x1 + (x_elements * y0)), xmask & ymask)
        tmp1 = tl.load(in_ptr0 + (y0 + (y_elements * x1)), xmask & ymask)
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr + (x1 + (x_elements * y0)), tmp2, xmask & ymask)

    @triton.jit
    def mul2_kernel(
        in_ptr0,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        output = 2 * x
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def mul2_inplace_kernel(
        ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(ptr + offsets, mask=mask)
        output = 2 * x
        tl.store(ptr + offsets, output, mask=mask)

    @triton.jit
    def zero_negs(x):
        return tl.where(x >= 0, x, 0)

    @triton.jit
    def indirection_kernel(
        in_ptr0,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
        ACTIVATION: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        if ACTIVATION == "mul2_inplace_kernel":
            mul2_inplace_kernel(in_ptr0, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        x = tl.load(in_ptr0 + offsets, mask=mask)
        tl.store(out_ptr + offsets, x, mask=mask)
