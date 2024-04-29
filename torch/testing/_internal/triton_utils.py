# mypy: ignore-errors

import unittest

from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.utils._triton import has_triton

requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")

if has_triton():
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
            triton.Config({"BLOCK_SIZE": 128}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE": 64}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=4),
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
                {"BLOCK_SIZE_X": 128, "BLOCK_SIZE_Y": 128}, num_stages=4, num_warps=4
            ),
            triton.Config(
                {"BLOCK_SIZE_X": 64, "BLOCK_SIZE_Y": 64}, num_stages=3, num_warps=8
            ),
            triton.Config(
                {"BLOCK_SIZE_X": 64, "BLOCK_SIZE_Y": 64}, num_stages=4, num_warps=4
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
    def add_kernel_with_scaling(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        scaling_factor,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = (x + y) * scaling_factor
        tl.store(out_ptr + offsets, output, mask=mask)

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
        elif ACTIVATION == "add_kernel":
            add_kernel(in_ptr0, in_ptr0, out_ptr, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        x = tl.load(in_ptr0 + offsets, mask=mask)
        tl.store(out_ptr + offsets, x, mask=mask)

    @triton.jit
    def double_strided_kernel(
        in_ptr,
        out_ptr,
        in_y_stride,
        out_y_stride,
        X_BLOCK_SIZE: "tl.constexpr",
        Y_BLOCK_SIZE: "tl.constexpr",
    ):
        xid = tl.program_id(axis=0)
        yid = tl.program_id(axis=1)
        x_start = xid * X_BLOCK_SIZE
        y_start = yid * Y_BLOCK_SIZE
        x_offsets = x_start + tl.arange(0, X_BLOCK_SIZE)
        y_offsets = y_start + tl.arange(0, Y_BLOCK_SIZE)
        src_offsets = y_offsets[:, None] * in_y_stride + x_offsets[None, :]
        dst_offsets = y_offsets[:, None] * out_y_stride + x_offsets[None, :]
        src = tl.load(in_ptr + src_offsets)
        tl.store(out_ptr + dst_offsets, src * 2.0)

    @triton.jit
    def inline_asm_kernel(X, Y, Z, n: "tl.constexpr", BLOCK: "tl.constexpr"):
        x = tl.load(X + tl.arange(0, BLOCK))
        y = tl.load(Y + tl.arange(0, BLOCK))
        s = tl.full([BLOCK], n, tl.int32)
        z = tl.inline_asm_elementwise(
            "shf.l.wrap.b32 $0, $1, $2, $3;",
            "=r,r, r, r",
            [x, y, s],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        tl.store(Z + tl.arange(0, BLOCK), z)

    @triton.jit
    def add_kernel_with_block_ptr(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        x = tl.load(
            tl.make_block_ptr(
                base=x_ptr,
                shape=[n_elements],
                strides=[1],
                offsets=[block_start],
                block_shape=[BLOCK_SIZE],
                order=[0],
            ),
            boundary_check=[0],
        )
        y = tl.load(
            tl.make_block_ptr(
                base=y_ptr,
                shape=[n_elements],
                strides=[1],
                offsets=[block_start],
                block_shape=[BLOCK_SIZE],
                order=[0],
            ),
            boundary_check=[0],
        )
        output = x + y
        tl.store(
            tl.make_block_ptr(
                base=output_ptr,
                shape=[n_elements],
                strides=[1],
                offsets=[block_start],
                block_shape=[BLOCK_SIZE],
                order=[0],
            ),
            output,
            boundary_check=[0],
        )

    @triton.jit
    def kernel_with_block_ptr_2d(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        x = tl.load(
            tl.make_block_ptr(
                base=x_ptr,
                shape=[n_elements, 1],
                strides=[1, 1],
                offsets=[block_start, 0],
                block_shape=[BLOCK_SIZE, 1],
                order=[1, 0],
            ),
            boundary_check=[0],
        )
        output = x
        tl.store(
            tl.make_block_ptr(
                base=output_ptr,
                shape=[n_elements, 1],
                strides=[1, 1],
                offsets=[block_start, 0],
                block_shape=[BLOCK_SIZE, 1],
                order=[1, 0],
            ),
            output,
            boundary_check=[0],
        )

    from triton.language import load, store

    @triton.jit
    def add_kernel_with_import(
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
        x = load(in_ptr0 + offsets, mask=mask)
        y = load(in_ptr1 + offsets, mask=mask)
        output = x + y
        store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def cond_op_kernel(
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
        if tl.program_id(0) == 0:
            output = x + y
        else:
            output = x * y
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def atomic_add_kernel(
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
        tl.atomic_add(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def add_4_times_kernel(
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
        for i in range(2):
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)
        i = 2
        while i > 0:
            i -= 1
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def add_kernel_out_of_order_fn2(
        in_ptr0,
        in_ptr1,
        n_elements,
        out_ptr,
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
