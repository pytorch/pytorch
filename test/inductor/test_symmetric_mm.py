# Owner(s): ["module: inductor", "module: optimizer"]

from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


class SymmetricMMTest(TestCase):
    def _check_symmetric_compile(self, fn, *args, uses_quack):
        torch._dynamo.reset()
        compiled = torch.compile(fn, fullgraph=True)
        stream = torch.cuda.Stream(device=args[0].device)
        stream.wait_stream(torch.cuda.current_stream(args[0].device))
        with torch.cuda.stream(stream):
            actual, code = run_and_get_code(compiled, *args)
            expected = fn(*args)
        stream.synchronize()
        check = FileCheck()
        if uses_quack:
            check.check("extern_kernels._quack_symmetric_mm")
        else:
            check.check_not("extern_kernels._quack_symmetric_mm")
        check.run("\n".join(code))
        self.assertEqual(actual, expected, rtol=2e-2, atol=1)
        return actual

    @skipIfNoCuteDSL
    @parametrize(
        "shape",
        [(4096, 4096), (5120, 8192), (2, 4096, 8192), (0, 4096, 4096)],
    )
    def test_quack_symmetric_mm(self, device, shape):
        if torch.cuda.get_device_capability(device)[0] != 10:
            self.skipTest("requires SM100")

        def fn(x):
            return torch.bmm(x, x.mT) if x.ndim == 3 else x @ x.T

        x = torch.randn(shape, device=device, dtype=torch.bfloat16)
        actual = self._check_symmetric_compile(fn, x, uses_quack=True)
        self.assertEqual(actual, actual.mT)

    @skipIfNoCuteDSL
    @parametrize(
        "case",
        [
            "alignment",
            "min_m",
            "aspect_ratio",
            "layout",
            "dtype",
            "not_transposed",
            "different_input",
        ],
    )
    def test_quack_symmetric_mm_fallback(self, device, case):
        if torch.cuda.get_device_capability(device)[0] != 10:
            self.skipTest("requires SM100")

        def fn(x):
            return x @ x.T

        args = None
        if case == "alignment":
            x = torch.randn((4097, 4104), device=device, dtype=torch.bfloat16)
        elif case == "min_m":
            x = torch.randn((4088, 4096), device=device, dtype=torch.bfloat16)
        elif case == "aspect_ratio":
            x = torch.randn((4104, 4096), device=device, dtype=torch.bfloat16)
        elif case == "layout":
            storage = torch.randn((4096, 8192), device=device, dtype=torch.bfloat16)
            x = storage[:, ::2]
        elif case == "dtype":
            x = torch.randn((4096, 4096), device=device, dtype=torch.float16)
        elif case == "not_transposed":

            def fn(x):
                return x @ x.permute(0, 1)

            x = torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)
        elif case == "different_input":

            def fn(x, y):
                return x @ y.T

            x = torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)
            args = (x, torch.randn_like(x))
        else:
            raise AssertionError(f"unexpected test case: {case}")

        self._check_symmetric_compile(fn, *(args or (x,)), uses_quack=False)

    @parametrize(
        "case", ["no_cute", "architecture", "hip", "cpp_wrapper", "fx_wrapper"]
    )
    def test_quack_symmetric_mm_environment_fallback(self, device, case):
        from torch._inductor.fx_passes.post_grad import _is_quack_symmetric_mm
        from torch._inductor.kernel.symmetric_mm import quack_symmetric_mm

        x = torch.empty((4096, 4096), device=device, dtype=torch.bfloat16)
        match = SimpleNamespace(
            kwargs={"x": SimpleNamespace(meta={"val": x}), "dims": [1, 0]}
        )
        with (
            mock.patch(
                "torch._inductor.fx_passes.post_grad.ensure_cute_available",
                return_value=case != "no_cute",
            ),
            mock.patch(
                "torch._inductor.utils.ensure_cute_available",
                return_value=case != "no_cute",
            ),
            mock.patch.object(
                torch.cuda,
                "get_device_capability",
                return_value=(11, 0) if case == "architecture" else (10, 0),
            ),
            mock.patch.object(torch.version, "hip", "6.3" if case == "hip" else None),
            config.patch(
                {
                    "cpp_wrapper": case == "cpp_wrapper",
                    "fx_wrapper": case == "fx_wrapper",
                }
            ),
        ):
            self.assertFalse(_is_quack_symmetric_mm(match))
            if case in ("no_cute", "architecture", "hip"):
                self.assertEqual(quack_symmetric_mm(x), x @ x.T)

    @skipIfNoCuteDSL
    @parametrize("case", ["unaligned_pointer", "contiguous_intermediate"])
    def test_quack_symmetric_mm_realizes_input(self, device, case):
        if torch.cuda.get_device_capability(device)[0] != 10:
            self.skipTest("requires SM100")

        if case == "unaligned_pointer":

            def fn(x):
                return x @ x.T

            storage = torch.randn(4096 * 4096 + 1, device=device, dtype=torch.bfloat16)
            x = storage[1:].view(4096, 4096)
        elif case == "contiguous_intermediate":

            def fn(x):
                y = x.permute(1, 0, 2).sum(2)
                return y @ y.T

            x = torch.randn((4096, 4096, 2), device=device, dtype=torch.bfloat16)
        else:
            raise AssertionError(f"unexpected test case: {case}")

        self._check_symmetric_compile(fn, x, uses_quack=True)

    def test_quack_grouped_symmetric_mm(self, device):
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("requires SM100 or SM110")

        from torch._vendor.quack.gemm_interface import gemm_symmetric_out

        x = torch.randn(2, 512, 1024, device=device, dtype=torch.bfloat16)
        gram = torch.empty(2, 512, 512, device=device, dtype=torch.bfloat16)
        gemm_symmetric_out(x, x.mT, gram)
        self.assertEqual(gram, torch.bmm(x, x.mT))

        update = torch.empty_like(gram)
        gemm_symmetric_out(gram, gram, update, C=gram, alpha=2.0315, beta=-4.775)
        expected = torch.baddbmm(gram, gram, gram, beta=-4.775, alpha=2.0315)
        self.assertEqual(update, expected, rtol=2e-2, atol=5e-1)
        self.assertEqual(update, update.mT)


instantiate_device_type_tests(SymmetricMMTest, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
