# Owner(s): ["module: linear algebra"]

import unittest
from itertools import product

import torch

from torch.quantization._quantized_conversions import (
    pack_int4_to_int8,
)

from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)

from torch.testing._internal.common_utils import (
    TestCase,
)

# Protects against includes accidentally setting the default dtype
assert torch.get_default_dtype() is torch.float32


class TestMixedDtypesLinearCpu(TestCase):
    @dtypes(torch.float16)  # , torch.bfloat16)
    def test_mixed_dtypes_linear(self, dtype: torch.dtype, device: str = "cpu"):
        def run_test(
            batch_shape,
            m,
            n,
            k,
            add_bias,
            activation,
            dtype,
            dtypeq,
            device,
            rtol,
            atol,
        ):
            val_lo, val_hi = -1, 1
            valq_lo, valq_hi = 0, 3
            input = make_tensor(
                *batch_shape, m, k, low=val_lo, high=val_hi, dtype=dtype, device=device
            )
            weight = make_tensor(
                n, k, low=valq_lo, high=valq_hi, dtype=dtypeq, device=device
            )
            scale = make_tensor(
                (n,), low=val_lo, high=val_hi, dtype=input.dtype, device=device
            )
            bias = (
                make_tensor(
                    (n,), low=val_lo, high=val_hi, dtype=input.dtype, device=device
                )
                if add_bias
                else None
            )

            input_ref = input.reshape(-1, input.shape[-1])

            # First, test plain multiplication.
            weight_ref = (weight.to(input.dtype) * scale.view(n, 1)).T
            weightq = pack_int4_to_int8(weight) if dtypeq == torch.quint4x2 else weight
            output_ref = torch.mm(input_ref, weight_ref).reshape(*input.shape[:-1], n)
            output = torch.ops.aten._mixed_dtypes_linear(
                input,
                weightq,
                scale,
            )
            torch.testing.assert_close(output, output_ref, rtol=rtol, atol=atol)

            # Second, test the linear operator itself.
            weight_ref = weight.to(input.dtype) * scale.view(n, 1)
            weightq = pack_int4_to_int8(weight) if dtypeq == torch.quint4x2 else weight
            output_ref = torch.nn.functional.linear(
                input_ref, weight_ref, bias=bias
            ).reshape(*input.shape[:-1], n)
            output = torch.ops.aten._mixed_dtypes_linear(
                input,
                weightq,
                scale,
                bias=bias,
                activation=activation,
            )
            torch.testing.assert_close(output, output_ref, rtol=rtol, atol=atol)

        dtypeqs = [torch.uint8]  # , torch.quint4x2] # FIXME!
        batch_shapes = [[], [2], [2, 1]]
        shapes = [
            [8, 64, 64],
            [8, 64, 128],
            [8, 128, 64],
            [8, 128, 128],
            [8, 128, 192],
            [8, 128, 256],
            [8, 256, 128],
            [8, 256, 384],
            [8, 384, 256],
        ]
        activations = [None]  # , "relu", "silu"] # FIXME!
        rtol, atol = 1e-3, 1e-3
        if dtype == torch.bfloat16:
            rtol, atol = 1e-2, 1e-3
        for dtypeq, batch_shape, (m, n, k), add_bias, activation in product(
            dtypeqs,
            batch_shapes,
            shapes,
            (False, True),
            activations,
        ):
            run_test(
                batch_shape,
                m,
                n,
                k,
                add_bias,
                activation,
                dtype,
                dtypeq,
                device,
                rtol,
                atol,
            )


instantiate_device_type_tests(TestMixedDtypesLinearCpu, globals(), except_for="cuda")

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
