# Owner(s): ["module: inductor"]


from unittest.mock import patch

import torch

from torch._inductor import config
from torch._inductor.fx_passes.pad_mm import (
    addmm_replace,
    bmm_replace,
    call_addmm,
    call_bmm,
    call_mm,
    get_alignment_size,
    mm_replace,
    should_pad_common,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CUDA


@config.patch({"shape_padding": True})
@instantiate_parametrized_tests
class PadMMTest(TestCase):
    @staticmethod
    def _check_tensor_alignment(tensor, expected_alignment):
        contiguous_dim_count = 0
        for stride in reversed(tensor.stride()):
            assert (stride == 1) or (
                stride % expected_alignment == 0
            ), f"Expected all non-contiguous strides of tensor with shape {tensor.shape} and strides {tensor.stride()} to be aligned to a multiple of {expected_alignment}"  # noqa: B950
            if stride == 1:
                contiguous_dim_count += 1

    @parametrize(
        "m,n,k", [(1, 1, 1), (16, 32, 64), (17, 33, 65), (16, 15, 8), (15, 32, 16)]
    )
    @parametrize("shape_pad_use_transpose", (False, True))
    def test_pad_nobatch(self, m=6, n=9, k=11, shape_pad_use_transpose: bool = True):
        with config.patch(
            {
                "shape_pad_use_transpose": shape_pad_use_transpose,
                "force_shape_pad": True,
            }
        ):
            mat1 = torch.ones((m, k), device="cuda", dtype=torch.float16)
            mat2 = torch.ones((k, n), device="cuda", dtype=torch.float16)
            bias = torch.ones((m, n), device="cuda", dtype=torch.float16)
            expected_alignment = get_alignment_size(mat1)
            assert (
                expected_alignment >= 2
            ), "Expected alignment should be greater or equal to 2"
            assert should_pad_common(
                mat1, mat2
            ), "This should pass the common padding criteria"
            assert should_pad_common(
                mat1, mat2, bias
            ), "This should pass the common padding criteria"

            orig_addmm = call_addmm
            called_checked = False

            def aten_addmm_checked(b, m1, m2, *args, **kwargs):
                nonlocal called_checked
                self._check_tensor_alignment(b, expected_alignment)
                self._check_tensor_alignment(m1, expected_alignment)
                self._check_tensor_alignment(m2, expected_alignment)
                res = orig_addmm(b, m1, m2, *args, **kwargs)
                self._check_tensor_alignment(res, expected_alignment)
                called_checked = True
                return res

            with patch(
                "torch._inductor.fx_passes.pad_mm.call_addmm", aten_addmm_checked
            ):
                addmm_result = addmm_replace(bias, mat1, mat2)

            addmm_expected_result = torch.addmm(bias, mat1, mat2)
            assert torch.allclose(
                addmm_result, addmm_expected_result
            ), "ADDMM results are not identical"

            addmm_compiled_result = torch.compile(
                lambda bias, mat1, mat2: torch.addmm(bias, mat1, mat2), dynamic=False
            )(bias, mat1, mat2)
            assert torch.allclose(
                addmm_compiled_result, addmm_expected_result
            ), "Compiled ADDMM results are not identical"
            self._check_tensor_alignment(addmm_compiled_result, expected_alignment)

            orig_mm = call_mm
            called_checked = False

            def aten_mm_checked(m1, m2, *args, **kwargs):
                nonlocal called_checked
                self._check_tensor_alignment(m1, expected_alignment)
                self._check_tensor_alignment(m2, expected_alignment)
                res = orig_mm(m1, m2, *args, **kwargs)
                self._check_tensor_alignment(res, expected_alignment)
                called_checked = True
                return res

            with patch("torch._inductor.fx_passes.pad_mm.call_mm", aten_mm_checked):
                mm_result = mm_replace(mat1, mat2)
            assert called_checked, "patched / checked aten.mm was not called at all"

            mm_expected_result = torch.mm(mat1, mat2)
            assert torch.allclose(
                mm_result, mm_expected_result
            ), "MM results are not identical"

            mm_compiled_result = torch.compile(lambda m1, m2: m1 @ m2, dynamic=False)(
                mat1, mat2
            )
            assert torch.allclose(
                mm_compiled_result, mm_expected_result
            ), "Compiled MM results are not identical"
            self._check_tensor_alignment(mm_compiled_result, expected_alignment)

    @parametrize(
        "m,n,k,batch_size",
        [
            (1, 1, 1, 8),
            (16, 32, 64, 8),
            (17, 33, 65, 7),
            (16, 33, 64, 4),
            (15, 32, 62, 3),
        ],
    )
    @parametrize("shape_pad_use_transpose", (False, True))
    def test_pad_batch(
        self, m=6, n=9, k=11, batch_size=3, shape_pad_use_transpose: bool = True
    ):
        with config.patch(
            {
                "shape_pad_use_transpose": shape_pad_use_transpose,
                "force_shape_pad": True,
            }
        ):
            mat1 = torch.ones((batch_size, m, k), device="cuda", dtype=torch.float16)
            mat2 = torch.ones((batch_size, k, n), device="cuda", dtype=torch.float16)
            expected_alignment = get_alignment_size(mat1)

            assert expected_alignment == 8, "Alignment for float16 should be 8"
            assert should_pad_common(
                mat1, mat2
            ), "This should pass the common padding criteria"

            orig_bmm = call_bmm
            called_checked = False

            def aten_bmm_checked(m1, m2, *args, **kwargs):
                nonlocal called_checked
                self._check_tensor_alignment(m1, expected_alignment)
                self._check_tensor_alignment(m2, expected_alignment)
                res = orig_bmm(m1, m2, *args, **kwargs)
                self._check_tensor_alignment(res, expected_alignment)
                called_checked = True
                return res

            with patch("torch._inductor.fx_passes.pad_mm.call_bmm", aten_bmm_checked):
                bmm_result = bmm_replace(mat1, mat2)

            bmm_expected_result = torch.bmm(mat1, mat2)

            assert torch.allclose(
                bmm_result, bmm_expected_result
            ), "BMM results are not identical"
            self._check_tensor_alignment(bmm_result, expected_alignment)

            bmm_compiled_result = torch.compile(
                lambda mat1, mat2: torch.bmm(mat1, mat2), dynamic=False
            )(mat1, mat2)
            assert torch.allclose(
                bmm_compiled_result, bmm_expected_result
            ), "Compiled BMM results are not identical"
            self._check_tensor_alignment(bmm_compiled_result, expected_alignment)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CUDA:
        run_tests()
