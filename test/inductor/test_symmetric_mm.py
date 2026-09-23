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
    def _check_grouped_muon(
        self,
        params,
        expected_plans=1,
        *,
        lr=1e-3,
        nesterov=True,
        ns_steps=5,
        adjust_lr_fn=None,
        compile_mode=None,
    ):
        from torch._inductor.kernel.muon import _PLAN_CACHE
        from torch.optim._muon import muon

        _PLAN_CACHE.clear()
        expected = [param.clone() for param in params]
        expected_bufs = [torch.zeros_like(param) for param in params]
        optimizer = torch.optim.Muon(
            params,
            lr=lr,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        stream = torch.cuda.Stream(device=params[0].device)
        torch._dynamo.reset()
        compiled_step = torch.compile(optimizer.step, fullgraph=True, mode=compile_mode)
        with torch.cuda.stream(stream):
            for _ in range(2):
                grads = [torch.randn_like(param) for param in params]
                for param, grad in zip(params, grads):
                    param.grad = grad
                compiled_step()

                with torch.no_grad():
                    muon(
                        expected,
                        grads,
                        expected_bufs,
                        lr=lr,
                        weight_decay=0.1,
                        momentum=0.95,
                        nesterov=nesterov,
                        ns_coefficients=(3.4445, -4.775, 2.0315),
                        ns_steps=ns_steps,
                        eps=1e-7,
                        adjust_lr_fn=adjust_lr_fn,
                        has_complex=False,
                    )
        stream.synchronize()
        self.assertEqual(params, expected, rtol=2e-2, atol=2e-2)
        plans = next(iter(_PLAN_CACHE.values()))
        self.assertEqual(
            sum(plan is not None for _, plan in plans.chunks), expected_plans
        )
        return plans

    def test_grouped_muon_rejects_inexact_normalization(self, device):
        from torch._inductor.kernel.muon import match_muon_foreach

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(8, 16, device=device, dtype=torch.bfloat16)
        denominator = graph.call_function(torch.ops.aten.clamp_min.default, (x, 1e-7))
        normalized = graph.call_function(torch.ops.aten.div.Tensor, (x, denominator))
        transpose = graph.call_function(
            torch.ops.aten.permute.default, (normalized, [1, 0])
        )
        gram = graph.call_function(torch.ops.aten.mm.default, (normalized, transpose))
        update = graph.call_function(
            torch.ops.aten.addmm.default,
            (gram, gram, gram),
            {"beta": -4.775, "alpha": 2.0315},
        )
        result = graph.call_function(
            torch.ops.aten.addmm.default,
            (normalized, update, normalized),
            {"beta": 3.4445},
        )
        graph.output(result)

        self.assertIsNone(match_muon_foreach(torch.fx.GraphModule({}, graph)))

    @skipIfNoCuteDSL
    @parametrize("shape", [(4096, 4096), (5120, 8192)])
    def test_quack_symmetric_mm(self, device, shape):
        if torch.cuda.get_device_capability(device)[0] != 10:
            self.skipTest("requires SM100")

        def fn(x):
            return x @ x.T

        x = torch.randn(shape, device=device, dtype=torch.bfloat16)
        torch._dynamo.reset()
        compiled = torch.compile(fn, fullgraph=True)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            actual, code = run_and_get_code(compiled, x)
            expected = fn(x)
        stream.synchronize()
        FileCheck().check("extern_kernels._quack_symmetric_mm").run(code[0])
        self.assertEqual(actual, expected)
        self.assertEqual(actual, actual.T)

    @skipIfNoCuteDSL
    def test_quack_batched_symmetric_mm(self, device):
        if torch.cuda.get_device_capability(device)[0] != 10:
            self.skipTest("requires SM100")

        def fn(x):
            return torch.bmm(x, x.mT)

        x = torch.randn((2, 4096, 8192), device=device, dtype=torch.bfloat16)
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        FileCheck().check("extern_kernels._quack_symmetric_mm").run(code[0])
        self.assertEqual(actual, fn(x))
        self.assertEqual(actual, actual.mT)

    @skipIfNoCuteDSL
    @parametrize(
        "case",
        ["k_alignment", "m_alignment", "min_m", "aspect_ratio", "layout", "dtype"],
    )
    def test_quack_symmetric_mm_eligibility_fallback(self, device, case):
        if torch.cuda.get_device_capability(device)[0] != 10:
            self.skipTest("requires SM100")

        def fn(x):
            return x @ x.T

        if case == "k_alignment":
            x = torch.randn((4096, 4097), device=device, dtype=torch.bfloat16)
        elif case == "m_alignment":
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
        else:
            raise AssertionError(f"unexpected test case: {case}")

        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        FileCheck().check_not("extern_kernels._quack_symmetric_mm").run(code[0])
        self.assertEqual(actual, fn(x), rtol=2e-2, atol=1)

    @skipIfNoCuteDSL
    @parametrize("case", ["not_transposed", "different_input"])
    def test_quack_symmetric_mm_near_match_fallback(self, device, case):
        if torch.cuda.get_device_capability(device)[0] != 10:
            self.skipTest("requires SM100")

        if case == "not_transposed":

            def fn(x, y):
                return x @ x.permute(0, 1)

        elif case == "different_input":

            def fn(x, y):
                return x @ y.T

        else:
            raise AssertionError(f"unexpected test case: {case}")

        x = torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)
        y = torch.randn_like(x)
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x, y)
        FileCheck().check_not("extern_kernels._quack_symmetric_mm").run(code[0])
        self.assertEqual(actual, fn(x, y), rtol=2e-2, atol=1)

    def test_quack_symmetric_mm_without_cute(self, device):
        from torch._inductor.fx_passes.post_grad import _is_quack_symmetric_mm

        x = torch.empty((4096, 4096), device=device, dtype=torch.bfloat16)
        match = SimpleNamespace(
            kwargs={"x": SimpleNamespace(meta={"val": x}), "dims": [1, 0]}
        )
        with mock.patch(
            "torch._inductor.fx_passes.post_grad.ensure_cute_available",
            return_value=False,
        ):
            self.assertFalse(_is_quack_symmetric_mm(match))

    @skipIfNoCuteDSL
    def test_quack_symmetric_mm_architecture_fallback(self, device):
        from torch._inductor.fx_passes.post_grad import _is_quack_symmetric_mm

        x = torch.empty((4096, 4096), device=device, dtype=torch.bfloat16)
        match = SimpleNamespace(
            kwargs={"x": SimpleNamespace(meta={"val": x}), "dims": [1, 0]}
        )
        with mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(11, 0)
        ):
            self.assertFalse(_is_quack_symmetric_mm(match))

    @skipIfNoCuteDSL
    def test_quack_symmetric_mm_hip_fallback(self, device):
        from torch._inductor.fx_passes.post_grad import _is_quack_symmetric_mm
        from torch._inductor.kernel.symmetric_mm import quack_symmetric_mm

        x = torch.empty((4096, 4096), device=device, dtype=torch.bfloat16)
        match = SimpleNamespace(
            kwargs={"x": SimpleNamespace(meta={"val": x}), "dims": [1, 0]}
        )
        with mock.patch.object(torch.version, "hip", "6.3"):
            self.assertFalse(_is_quack_symmetric_mm(match))
            self.assertEqual(quack_symmetric_mm(x), x @ x.T)

    @skipIfNoCuteDSL
    def test_quack_symmetric_mm_unaligned_pointer_fallback(self, device):
        if torch.cuda.get_device_capability(device)[0] != 10:
            self.skipTest("requires SM100")

        def fn(x):
            return x @ x.T

        storage = torch.randn(4096 * 4096 + 1, device=device, dtype=torch.bfloat16)
        x = storage[1:].view(4096, 4096)
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        FileCheck().check("extern_kernels._quack_symmetric_mm").run(code[0])
        self.assertEqual(actual, fn(x), rtol=2e-2, atol=1)

    @skipIfNoCuteDSL
    def test_quack_symmetric_mm_contiguous_intermediate(self, device):
        if torch.cuda.get_device_capability(device)[0] != 10:
            self.skipTest("requires SM100")

        def fn(x):
            y = x.permute(1, 0, 2).sum(2)
            return y @ y.T

        x = torch.randn((4096, 4096, 2), device=device, dtype=torch.bfloat16)
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        FileCheck().check("extern_kernels._quack_symmetric_mm").run(code[0])
        self.assertEqual(actual, fn(x))

    @skipIfNoCuteDSL
    def test_quack_symmetric_mm_empty_batch(self, device):
        if torch.cuda.get_device_capability(device)[0] != 10:
            self.skipTest("requires SM100")

        def fn(x):
            return torch.bmm(x, x.mT)

        x = torch.randn((0, 4096, 4096), device=device, dtype=torch.bfloat16)
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        FileCheck().check("extern_kernels._quack_symmetric_mm").run(code[0])
        self.assertEqual(actual, fn(x))

    @skipIfNoCuteDSL
    @parametrize("wrapper", ["cpp_wrapper", "fx_wrapper"])
    def test_quack_symmetric_mm_wrapper_fallback(self, device, wrapper):
        if torch.cuda.get_device_capability(device)[0] != 10:
            self.skipTest("requires SM100")

        def fn(x):
            return x @ x.T

        x = torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)
        with config.patch({wrapper: True}):
            actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        FileCheck().check_not("quack_symmetric_mm").run("\n".join(code))
        self.assertEqual(actual, fn(x))

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

    def test_quack_pointer_grouped_symmetric_mm(self, device):
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("requires SM100 or SM110")

        from torch._inductor.kernel.grouped_symmetric_mm import GroupedSymmetricPlan

        inputs = [
            torch.randn(512, 1024, device=device, dtype=torch.bfloat16),
            torch.randn(768, 1536, device=device, dtype=torch.bfloat16),
        ]
        gram_plan = GroupedSymmetricPlan(inputs)
        with self.assertRaisesRegex(ValueError, "compiled.*incompatible"):
            GroupedSymmetricPlan(inputs[:1], compiled_plan=gram_plan)
        grams = gram_plan()
        for x, gram in zip(inputs, grams):
            self.assertEqual(gram, torch.mm(x, x.T))

        reuse_inputs = [
            torch.randn(x.shape[0], 2048, device=device, dtype=torch.bfloat16)
            for x in inputs
        ]
        reuse_plan = GroupedSymmetricPlan(reuse_inputs, compiled_plan=gram_plan)
        self.assertIs(reuse_plan.compiled, gram_plan.compiled)
        for x, gram in zip(reuse_inputs, reuse_plan()):
            self.assertEqual(gram, torch.mm(x, x.T))

        update_plan = GroupedSymmetricPlan(grams, c=grams, alpha=2.0315, beta=-4.775)
        updates = update_plan()
        for gram, update in zip(grams, updates):
            expected = torch.addmm(gram, gram, gram, beta=-4.775, alpha=2.0315)
            self.assertEqual(update, expected, rtol=2e-2, atol=5e-1)
            self.assertEqual(update, update.T)

    def test_quack_grouped_symmetric_mm_rejects_unaligned_inputs(self, device):
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("requires SM100 or SM110")

        from torch._inductor.kernel.grouped_symmetric_mm import GroupedSymmetricPlan

        inputs = [
            torch.randn(127, 1024, device=device, dtype=torch.bfloat16),
            torch.randn(128, 1025, device=device, dtype=torch.bfloat16),
            torch.randn(128 * 1024 + 1, device=device, dtype=torch.bfloat16)[1:].view(
                128, 1024
            ),
        ]
        for x in inputs:
            with self.assertRaisesRegex(ValueError, "aligned contiguous"):
                GroupedSymmetricPlan([x])

    def test_muon_uses_grouped_symmetric_mm(self, device):
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("requires SM100 or SM110")

        params = [
            torch.randn(
                (1536, 5120) if index % 2 == 0 else (5120, 1536),
                device=device,
                dtype=torch.bfloat16,
            )
            for index in range(4)
        ]
        plans = self._check_grouped_muon(params)
        plan = next(plan for _, plan in plans.chunks if plan is not None)
        self.assertIs(plan.gram_plans[0].compiled, plan.update_plan.compiled)

    def test_muon_foreach_map_compile(self, device):
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("requires SM100 or SM110")

        from torch.optim._muon import muon

        params = [
            torch.randn(512, 1024, device=device, dtype=torch.bfloat16)
            for _ in range(4)
        ]
        grads = [torch.randn_like(param) for param in params]
        bufs = [torch.zeros_like(param) for param in params]
        expected = [param.clone() for param in params]
        expected_bufs = [buf.clone() for buf in bufs]

        def step(step_params, step_grads, step_bufs, foreach):
            muon(
                step_params,
                step_grads,
                step_bufs,
                foreach=foreach,
                lr=1e-3,
                weight_decay=0.1,
                momentum=0.95,
                nesterov=True,
                ns_coefficients=(3.4445, -4.775, 2.0315),
                ns_steps=5,
                eps=1e-7,
                adjust_lr_fn=None,
                has_complex=False,
            )

        step(expected, grads, expected_bufs, False)
        torch.compile(step, fullgraph=True)(params, grads, bufs, True)
        self.assertEqual(params, expected, rtol=2e-2, atol=2e-2)
        self.assertEqual(bufs, expected_bufs)

    def test_muon_merges_sparse_small_symmetric_mm(self, device):
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("requires SM100 or SM110")

        params = [
            torch.randn(m, 1024, device=device, dtype=torch.bfloat16)
            for m in (128, 256, 512)
            for _ in range(4)
        ]
        self._check_grouped_muon(params)

    def test_muon_groups_heterogeneous_symmetric_mm(self, device):
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("requires SM100 or SM110")

        params = []
        for k in (5120, 7168):
            params.extend(
                torch.randn(1536, k, device=device, dtype=torch.bfloat16)
                for _ in range(4)
            )
        self._check_grouped_muon(params)

    def test_muon_unaligned_shape_fallback(self, device):
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("requires SM100 or SM110")

        params = [
            torch.randn(127, 1025, device=device, dtype=torch.bfloat16)
            for _ in range(4)
        ]
        self._check_grouped_muon(params, expected_plans=0)

    def test_muon_cudagraph_fallback(self, device):
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("requires SM100 or SM110")

        params = [
            torch.randn(128, 1024, device=device, dtype=torch.bfloat16)
            for _ in range(4)
        ]
        self._check_grouped_muon(params, compile_mode="reduce-overhead")

    def test_muon_empty_gradients(self, device):
        param = torch.randn(8, 16, device=device, requires_grad=True)
        optimizer = torch.optim.Muon([param])
        torch.compile(optimizer.step, fullgraph=True)()

    def test_muon_zero_steps(self, device):
        param = torch.randn(8, 16, device=device, requires_grad=True)
        expected = param.detach().clone().requires_grad_()
        grad = torch.randn_like(param)
        param.grad = grad
        expected.grad = grad.clone()
        optimizer = torch.optim.Muon([param], ns_steps=0)
        expected_optimizer = torch.optim.Muon([expected], ns_steps=0)
        expected_optimizer.step()
        torch.compile(optimizer.step, fullgraph=True)()
        self.assertEqual(param, expected)

    def test_muon_foreach_validates_steps(self, device):
        from torch.optim._muon import muon

        param = torch.randn(8, 16, device=device)
        grad = torch.randn_like(param)
        buf = torch.zeros_like(param)
        with self.assertRaisesRegex(ValueError, "less than 100"):
            muon(
                [param],
                [grad],
                [buf],
                foreach=True,
                lr=1e-3,
                weight_decay=0.1,
                momentum=0.95,
                nesterov=True,
                ns_coefficients=(3.4445, -4.775, 2.0315),
                ns_steps=100,
                eps=1e-7,
                adjust_lr_fn=None,
                has_complex=False,
            )

    def test_muon_uses_singleton_symmetric_mm(self, device):
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("requires SM100 or SM110")

        params = [torch.randn(4096, 8192, device=device, dtype=torch.bfloat16)]
        self._check_grouped_muon(params)

    @parametrize("count", [2, 4])
    def test_muon_uses_direct_symmetric_mm(self, device, count):
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("requires SM100 or SM110")

        params = [
            torch.randn(
                (4096, 8192) if index % 2 == 0 else (8192, 4096),
                device=device,
                dtype=torch.bfloat16,
            )
            for index in range(count)
        ]
        self._check_grouped_muon(params)

    def test_muon_groups_small_symmetric_mm(self, device):
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("requires SM100 or SM110")

        params = [
            torch.randn(m, 1024, device=device, dtype=torch.bfloat16)
            for m in (128, 256)
            for _ in range(10)
        ]
        self._check_grouped_muon(params, expected_plans=2)

    @parametrize("dtype", [torch.float32, torch.float64])
    @parametrize("contiguous", [True, False])
    def test_muon_grouped_non_bfloat16_without_nesterov(
        self, device, dtype, contiguous
    ):
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("requires SM100 or SM110")

        params = []
        for _ in range(4):
            param = torch.randn(128, 1024, device=device, dtype=dtype)
            params.append(param if contiguous else param.T.contiguous().T)
        self._check_grouped_muon(params, expected_plans=0, nesterov=False, ns_steps=4)

    @parametrize("lr_device", ["cpu", "cuda"])
    def test_muon_grouped_tensor_lr(self, device, lr_device):
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("requires SM100 or SM110")

        params = [
            torch.randn(128, 1024, device=device, dtype=torch.bfloat16)
            for _ in range(4)
        ]
        lr = torch.tensor(1e-3, device=lr_device)
        self._check_grouped_muon(params, lr=lr)

    @parametrize("adjust_lr_fn", ["original", "match_rms_adamw", "spectral_unclamped"])
    def test_muon_grouped_adjust_lr(self, device, adjust_lr_fn):
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("requires SM100 or SM110")

        params = [
            torch.randn(
                (1536, 5120) if index % 2 == 0 else (5120, 1536),
                device=device,
                dtype=torch.bfloat16,
            )
            for index in range(4)
        ]
        self._check_grouped_muon(params, adjust_lr_fn=adjust_lr_fn)

    def test_muon_grouped_requires_cutedsl(self, device):
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("requires SM100 or SM110")

        params = [
            torch.randn(128, 1024, device=device, dtype=torch.bfloat16)
            for _ in range(4)
        ]
        from torch._inductor.kernel.muon import _PLAN_CACHE

        _PLAN_CACHE.clear()
        with mock.patch(
            "torch._inductor.utils.ensure_cute_available", return_value=False
        ):
            plans = self._check_grouped_muon(params, expected_plans=0)
        self.assertTrue(all(plan is None for _, plan in plans.chunks))


instantiate_device_type_tests(SymmetricMMTest, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
