# Owner(s): ["module: inductor"]
import math
import os
import sys

import torch
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.test_operators import realize
from torch._inductor.utils import fresh_cache, is_big_gpu, run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import slowTest
from torch.testing._internal.inductor_utils import (
    get_func_call,
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU_AND_TRITON,
    IS_BIG_GPU,
)
from torch.utils._triton import has_triton_cuda_tma_device


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

import contextlib
import unittest
from types import SimpleNamespace

from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model,
    check_model_gpu,
    copy_tests,
    skip_if_cpp_wrapper,
    skip_if_device,
)
from torch._inductor import config
from torch._inductor.choices import (
    InductorChoices,
    MultiTemplateFusionDecision,
    register_inductor_choices,
    registered_inductor_choices,
    unregister_inductor_choices,
)
from torch._inductor.ir import MultiTemplateBuffer
from torch._inductor.scheduler import Scheduler
from torch._inductor.select_algorithm import (
    ExternKernelCaller,
    TritonTemplate,
    TritonTemplateCaller,
)
from torch._inductor.virtualized import _choices, threadlocal


_recording_ids = iter(range(1 << 30))


class RecordingFusionChoices(InductorChoices):
    """Decides each MultiTemplateBuffer fusion with ``decide(call)`` and records
    the call: the buffer, the buffer names of the other side, whether it is an
    epilogue, the template the buffer was finalized to before the call, whether
    the buffer was marked as having a fused device assert, and the decision."""

    def __init__(self, decide=None):
        super().__init__()
        self.decide = decide or (lambda call: None)
        self.calls = []
        # Each policy its own cache key: a compile never reuses a graph that
        # another policy decided.
        self.id = next(_recording_ids)

    def uuid(self):
        return f"recording-fusion-{self.id}"

    def multi_template_fusion_decision(
        self, scheduler, multi_node, other_nodes, epilogue
    ):
        call = SimpleNamespace(
            multi_node=multi_node,
            other_buffers=[n for node in other_nodes for n in node.get_buffer_names()],
            epilogue=epilogue,
            finalized=multi_node._render_caller,
            marked_unsafe=multi_node.get_name()
            in scheduler.templates_with_unsafe_asserts,
        )
        call.decision = self.decide(call)
        self.calls.append(call)
        return call.decision


class _OtherChoices(InductorChoices):
    """A second contributor that implements no fusion decision."""

    def uuid(self):
        return "other"


@contextlib.contextmanager
def _choices_from_config():
    """V.choices is created from config.inductor_choices_class and the
    registered contributors on first use, then cached for the thread; drop the
    cached handler so a compile resolves it again."""
    prior = getattr(threadlocal, _choices._key, None)
    if prior is not None:
        delattr(threadlocal, _choices._key)
    try:
        yield
    finally:
        if hasattr(threadlocal, _choices._key):
            delattr(threadlocal, _choices._key)
        if prior is not None:
            setattr(threadlocal, _choices._key, prior)


def _triton_choices(multi_node):
    return [c for c in multi_node.choices if isinstance(c, TritonTemplateCaller)]


def _accept(index=0):
    """Accepts with the buffer's ``index``-th Triton template choice."""
    return lambda call: MultiTemplateFusionDecision(
        fuse=True, choice=_triton_choices(call.multi_node)[index]
    )


class TestCase(InductorTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "benchmark_kernel": True,
                    "benchmark_fusion": True,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()


class BenchmarkFusionTestTemplate:
    def test_softmax(self):
        def f(x):
            return torch.nn.functional.softmax(x, dim=-1)

        self.common(f, (torch.rand(2, 8192),))

    @slowTest
    def test_resnet18(self):
        try:
            import torchvision
        except ImportError:
            self.skipTest("TorchVision not available")

        model = torchvision.models.resnet18()
        model.eval()
        batch_size = 16
        inputs = (torch.randn((batch_size, 3, 224, 224)),)
        self.common(model, inputs, atol=1e-2, rtol=1e-2)

    def test_register_spills(self):
        """
        The test can potentially trigger register spills
        """
        old_benchmark_fn = Scheduler.benchmark_fused_nodes

        def new_benchmark_fn(scheduler, nodes):
            """
            We override Scheduler.benchmark_fused_nodes to return latency 1.0
            if there are no register spills. Without this, we may not able to
            test the code path handling register spilling because before register
            start spilling, the related fusion may have already been skipped
            due to longer lantency.
            """
            ms, path = old_benchmark_fn(scheduler, nodes)
            if not math.isinf(ms):
                ms = 1.0
            return ms, path

        # Disable dynamic_scale_rblock to make it easier to trigger register
        # spilling.
        with (
            unittest.mock.patch.object(
                Scheduler, "benchmark_fused_nodes", new_benchmark_fn
            ),
            config.patch("dynamic_scale_rblock", False),
        ):
            S = 512

            def f(*inputs):
                inputs = list(inputs)
                outputs = []
                out = torch.zeros(S, device=self.device)
                for x in inputs:
                    x = x * 2
                    x = x + 1
                    x = x.sum(dim=-1)
                    outputs.append(x)
                    out = out + x
                return outputs, out

            N = int(os.environ.get("NINP", "30"))
            inputs = [torch.randn(S, 2560, device=self.device) for _ in range(N)]
            opt_f = torch.compile(f)
            opt_f(*inputs)

    def test_foreach_kernel(self):
        """
        Benchmark fusion should skip benchmarking kernels involves foreach kernel
        for now. Without the skipping logic, `codegen_node_schedule` may fail.
        """
        a = torch.randn(1024, 256, device=self.device)
        b = torch.randn(1024, 512, device=self.device)

        def f(a, b):
            a, b = torch._foreach_abs([a, b])
            return a + 1, b + 2

        self.common(f, (a, b))

    @skip_if_device(lambda device: device != GPU_TYPE, "GPU only")
    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    @config.patch(max_autotune_gemm_backends="TRITON")
    def test_avoid_register_spilling(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("GPU only")

        from torch.nn.functional import gelu

        def foo(m, inp):
            curr = m(inp)
            tmps = []
            for _ in range(4):
                curr = gelu(curr)
                for t in tmps:
                    curr = curr + t
                tmps.append(curr)

            return curr

        m = torch.nn.Linear(2048, 2048, bias=True).half().to(GPU_TYPE)
        inp = torch.rand([2048, 2048]).half().to(GPU_TYPE)

        with torch.no_grad():
            foo_c = torch.compile(mode="max-autotune-no-cudagraphs")(foo)

            _, out_code = run_and_get_code(foo_c, m, inp)

            # occasionally, CI will make this one kernel. just skip in this case
            if out_code[0].count("def triton_") != 2:
                return

            # should be multiple triton invocations
            FileCheck().check("async_compile.wait").check_count(
                ".run", 2, exactly=True
            ).run(out_code[0])

        with (
            config.patch({"benchmark_fusion": False, "epilogue_fusion": False}),
            torch.no_grad(),
        ):
            torch._dynamo.reset()

            foo_c = torch.compile(mode="max-autotune-no-cudagraphs")(foo)

            _, out_code2 = run_and_get_code(foo_c, m, inp)

        for c in out_code[0], out_code2[0]:
            FileCheck().check("async_compile.wait").check("DeviceGuard").check_count(
                f"empty_strided_{GPU_TYPE}", 1, exactly=True
            ).check_regex("buf[0-9]* = buf[0-9]*; del buf[0-9]*").check("return").run(c)

    def test_tield_kernel_fusion(self):
        def f(x):
            y = realize(x + x.t())
            return y + 1

        x = torch.randn(1024, 1024, device=self.device)
        # Disable lowp check due to non-deterministic kernel fusion strategies affecting fp16 precision.
        self.common(f, (x,), check_lowp=False)


if HAS_GPU_AND_TRITON:

    class BenchmarkFusionGpuTest(TestCase):
        common = check_model_gpu
        device = GPU_TYPE

    copy_tests(BenchmarkFusionTestTemplate, BenchmarkFusionGpuTest, GPU_TYPE)

    class BenchmarkingTest(TestCase):
        @unittest.skipIf(
            getattr(torch, GPU_TYPE).device_count() < 2,
            "The test need at least 2 devices",
        )
        @skip_if_cpp_wrapper("This tests triton scheduling directly")
        def test_benchmark_on_non_zero_device(self):
            hit_count = 0
            with getattr(torch, GPU_TYPE).device(f"{GPU_TYPE}:0"):

                @torch.compile
                def relu(x):
                    return realize(x.relu()) + x

                x = torch.randn(int(16e6), device=f"{GPU_TYPE}:1")

                orig_benchmark_codegened_module = (
                    TritonScheduling.benchmark_codegened_module
                )

                def benchmark_codegened_module(*args, **kwargs):
                    nonlocal hit_count
                    hit_count += 1
                    ms, path = orig_benchmark_codegened_module(*args, **kwargs)
                    self.assertTrue(ms > 0)
                    return ms, path

                with unittest.mock.patch.object(
                    TritonScheduling,
                    "benchmark_codegened_module",
                    benchmark_codegened_module,
                ):
                    relu(x)
                self.assertTrue(hit_count > 0)

    class BenchmarkMultiTemplateFusionGpuTest(InductorTestCase):
        @classmethod
        def setUpClass(cls):
            super().setUpClass()
            cls._stack = contextlib.ExitStack()
            cls._stack.enter_context(
                config.patch(
                    {
                        "benchmark_kernel": True,
                        "benchmark_fusion": True,
                        "benchmark_template_fusion": True,
                    }
                )
            )

        @classmethod
        def tearDownClass(cls):
            cls._stack.close()
            super().tearDownClass()

        def setUp(self):
            super().setUp()
            if not is_big_gpu():
                return self.skipTest("Need a big GPU to run max_autotune=True")

        def _equivalent_output_code_impl(self, size, first_dim=None, activation=True):
            def foo(m, inp):
                a = m(inp)
                if activation:
                    return torch.nn.functional.relu(a)
                return a

            foo_c = torch.compile(mode="max-autotune-no-cudagraphs")(foo)
            first_dim = first_dim if first_dim is not None else size

            m = torch.nn.Linear(size, size, bias=True).half().to(GPU_TYPE)
            inp = torch.rand([first_dim, size]).half().to(GPU_TYPE)

            with torch.no_grad():
                res, code = run_and_get_code(foo_c, m, inp)

            torch._dynamo.reset()
            with config.patch(benchmark_template_fusion=False):
                foo_c = torch.compile(mode="max-autotune-no-cudagraphs")(foo)
                with torch.no_grad():
                    res2, code2 = run_and_get_code(foo_c, m, inp)

            self.assertEqual(res, res2, atol=1e-4, rtol=1.1)
            return code, code2

        @fresh_cache()
        @config.patch(
            {
                "max_autotune_gemm_backends": "TRITON",
                "benchmark_template_fusion": False,
            }
        )
        def test_equivalent_template_code(self):
            code, code2 = self._equivalent_output_code_impl(256)
            for out_code in [code, code2]:
                FileCheck().check(get_func_call()).check_count(
                    "empty_strided", 1, exactly=True
                ).check("triton_tem_fused_addmm_relu_t_0").check_count(
                    ".reset()" if config.cpp_wrapper else "del", 3, exactly=True
                ).check("" if config.cpp_wrapper else "return").run(out_code[0])

        @fresh_cache()
        @config.patch(max_autotune_gemm_backends="ATEN")
        def test_equivalent_extern_code(self):
            torch._dynamo.reset()

            code, code2 = self._equivalent_output_code_impl(512, 1, False)

            for out_code in [code, code2]:
                FileCheck().check(get_func_call()).check_count(
                    "empty_strided", 1, exactly=True
                ).check("" if config.cpp_wrapper else "extern_kernels.").check_count(
                    ".reset()" if config.cpp_wrapper else "del", 3, exactly=True
                ).check("" if config.cpp_wrapper else "return").run(out_code[0])

        def test_changed_layout(self):
            # cat addmm planning will change layout - make sure propagated
            def fn(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
                return torch.cat(
                    [
                        torch.addmm(a, b, c),
                        torch.addmm(b, c, a),
                    ],
                    1,
                )

            args = [
                torch.randn(4, 4, device=GPU_TYPE),
                torch.randn(4, 4, device=GPU_TYPE),
                torch.randn(4, 4, device=GPU_TYPE),
            ]

            expected = fn(*args)
            actual = torch.compile(fn, mode="max-autotune")(*args)
            self.assertEqual(expected, actual)

            torch._dynamo.reset()

        def _check_tma_store_view_epilogue_fuses(self):
            # TMA-store choices tile the output 2D over (M, N). This epilogue
            # views the (384, 512) output as (256, 3, 256), which that tiling
            # can't split, so fusing it into a TMA-store choice raises CantSplit
            # while the other choices fuse fine.
            def f(a, b, bias):
                return (a @ b).view(256, 3, 256) * bias.view(3, 1)

            a = torch.randn(384, 256, device=GPU_TYPE, dtype=torch.float16)
            b = torch.randn(256, 512, device=GPU_TYPE, dtype=torch.float16)
            bias = torch.randn(3, device=GPU_TYPE, dtype=torch.float16)

            out, code = run_and_get_code(torch.compile(f), a, b, bias)
            self.assertEqual(out, f(a, b, bias), atol=1e-1, rtol=1e-2)
            FileCheck().check("triton_tem_fused").check_count(
                "@triton.jit", 1, exactly=True
            ).run(code[0])

        @unittest.skipIf(
            not has_triton_cuda_tma_device(), "Needs TMA-store template choices"
        )
        @fresh_cache()
        @config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_template_tma_store": True,
                "multi_kernel_hints": [64],
            }
        )
        def test_multi_kernel_hints_skips_cant_split_choice(self):
            self._check_tma_store_view_epilogue_fuses()

        @unittest.skipIf(
            not has_triton_cuda_tma_device(), "Needs TMA-store template choices"
        )
        @fresh_cache()
        @config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_template_tma_store": True,
                "max_template_fusion_benchmarked_choices": 1,
            }
        )
        def test_cant_split_choice_not_counted_as_benchmarked(self):
            # Rank the TMA-store choices first, so the one benchmarked slot goes
            # to a choice whose fused kernel raises CantSplit.
            orig_generate = TritonTemplate.generate
            orig_choice_timings = MultiTemplateBuffer.choice_timings
            tma_store_choices = []

            def generate(self, *args, **kwargs):
                choice = orig_generate(self, *args, **kwargs)
                if choice is not None and kwargs.get("tma_store", False):
                    choice._test_tma_store = True
                    tma_store_choices.append(choice)
                return choice

            def choice_timings(self, hint_override=None):
                timings = orig_choice_timings(self, hint_override)
                ranked = sorted(timings.items(), key=lambda kv: kv[1])
                return {
                    c: 1.0 if getattr(c, "_test_tma_store", False) else 1.0 + 1e-6 * i
                    for i, (c, _) in enumerate(ranked, 1)
                }

            with (
                unittest.mock.patch.object(TritonTemplate, "generate", generate),
                unittest.mock.patch.object(
                    MultiTemplateBuffer, "choice_timings", choice_timings
                ),
            ):
                self._check_tma_store_view_epilogue_fuses()
            self.assertTrue(tma_store_choices)

    class MultiTemplateFusionDecisionGpuTest(InductorTestCase):
        """V.choices.multi_template_fusion_decision in real compilations: the
        hook alone decides whether a MultiTemplateBuffer fuses and the
        template it is finalized to."""

        @classmethod
        def setUpClass(cls):
            super().setUpClass()
            cls._stack = contextlib.ExitStack()
            cls._stack.enter_context(
                config.patch(
                    {
                        "max_autotune": True,
                        "max_autotune_gemm_backends": "TRITON",
                        "benchmark_template_fusion": True,
                        "prologue_fusion": True,
                        "shape_padding": False,
                        "test_configs.max_mm_configs": 4,
                    }
                )
            )

        @classmethod
        def tearDownClass(cls):
            cls._stack.close()
            super().tearDownClass()

        def setUp(self):
            super().setUp()
            # After the base setUp, which enables the FX graph cache: every
            # compile runs the scheduler and so the hook.
            no_graph_cache = config.patch(
                {"fx_graph_cache": False, "fx_graph_remote_cache": False}
            )
            no_graph_cache.__enter__()
            self.addCleanup(no_graph_cache.__exit__, None, None, None)
            if not is_big_gpu():
                return self.skipTest("Need a big GPU to run max_autotune=True")
            torch._dynamo.reset()

        def compile(self, fn, *args, choices=None):
            """The output code of ``fn`` compiled with ``choices`` installed
            (or without an override), checked against eager."""
            torch._dynamo.reset()
            with (
                _choices_from_config(),
                config.patch(inductor_choices_class=lambda: choices)
                if choices is not None
                else contextlib.nullcontext(),
            ):
                out, code = run_and_get_code(torch.compile(fn), *args)
            self.assertEqual(out, fn(*args), atol=5e-2, rtol=1e-2)
            return code[0]

        def assert_kernels(self, code, count):
            FileCheck().check_count("@triton.jit", count, exactly=True).run(code)

        @staticmethod
        def _mm_relu(a, b):
            return (a @ b).relu()

        @staticmethod
        def _upcast_mm(x, y):
            return x.to(y.dtype) @ y

        @staticmethod
        def _upcast_mm_relu(x, y):
            return (x.to(y.dtype) @ y).relu()

        def _fp16(self, *shape):
            return torch.randn(*shape, device=GPU_TYPE, dtype=torch.float16)

        def _upcast_args(self):
            return (
                self._fp16(256, 256),
                torch.randn(256, 256, device=GPU_TYPE, dtype=torch.float32),
            )

        @fresh_cache()
        def test_accept_and_decline_an_epilogue(self):
            args = (self._fp16(256, 256), self._fp16(256, 256))

            # Installed by registration next to another contributor: the
            # composed handler dispatches the hook to its only implementer.
            accept = RecordingFusionChoices(_accept())
            prior = registered_inductor_choices()
            for key, _ in prior:
                unregister_inductor_choices(key)
            try:
                register_inductor_choices("other", _OtherChoices)
                register_inductor_choices("recording", lambda: accept)
                with config.patch(inductor_choices_class=None):
                    code = self.compile(self._mm_relu, *args)
            finally:
                for key, _ in registered_inductor_choices():
                    unregister_inductor_choices(key)
                for key, factory in prior:
                    register_inductor_choices(key, factory)
            self.assert_kernels(code, 1)
            FileCheck().check("triton_tem_fused_mm_relu").run(code)
            [call] = accept.calls
            self.assertTrue(call.epilogue)
            self.assertIsNone(call.finalized)
            self.assertIs(call.multi_node._render_caller, call.decision.choice)

            decline = RecordingFusionChoices(
                lambda call: MultiTemplateFusionDecision(fuse=False)
            )
            code = self.compile(self._mm_relu, *args, choices=decline)
            self.assert_kernels(code, 2)
            FileCheck().check("triton_poi_fused_relu").run(code)
            self.assertTrue(decline.calls)
            self.assertTrue(all(call.epilogue for call in decline.calls))

        @fresh_cache()
        def test_a_hook_returning_none_keeps_the_default_decision(self):
            args = (self._fp16(256, 256), self._fp16(256, 256))
            default = self.compile(self._mm_relu, *args)
            consulted = RecordingFusionChoices()
            code = self.compile(self._mm_relu, *args, choices=consulted)
            self.assertTrue(consulted.calls)
            # The benchmark path fuses the relu, as without an override.
            for out in (default, code):
                self.assert_kernels(out, 1)
                FileCheck().check("triton_tem_fused_mm_relu").run(out)

        @fresh_cache()
        def test_a_producer_fuses_with_the_chosen_template(self):
            choices = RecordingFusionChoices(_accept())
            code = self.compile(self._upcast_mm, *self._upcast_args(), choices=choices)
            self.assert_kernels(code, 1)
            [call] = choices.calls
            self.assertFalse(call.epilogue)
            self.assertIs(call.multi_node._render_caller, call.decision.choice)

        @fresh_cache()
        def test_successive_fusions_keep_the_finalized_template(self):
            def first_then(later):
                def decide(call):
                    if call.finalized is None:
                        return _accept()(call)
                    return later(call)

                return decide

            # Without a choice, a later fusion reuses the finalized template.
            reuse = RecordingFusionChoices(
                first_then(lambda call: MultiTemplateFusionDecision(fuse=True))
            )
            code = self.compile(
                self._upcast_mm_relu, *self._upcast_args(), choices=reuse
            )
            self.assertEqual(len(reuse.calls), 2)
            first, second = reuse.calls
            self.assertIs(second.finalized, first.decision.choice)
            self.assertEqual({first.epilogue, second.epilogue}, {True, False})
            self.assert_kernels(code, 1)

            # Another template for a later fusion is declined; the buffer keeps
            # its template.
            conflict = RecordingFusionChoices(first_then(_accept(1)))
            code = self.compile(
                self._upcast_mm_relu, *self._upcast_args(), choices=conflict
            )
            first = conflict.calls[0]
            self.assertTrue(len(conflict.calls) >= 2)
            self.assertIs(first.multi_node._render_caller, first.decision.choice)
            self.assert_kernels(code, 2)

        def _benchmarks(self, events):
            """Appends the name of each Scheduler benchmark call to ``events``."""
            stack = contextlib.ExitStack()
            for name in ("benchmark_fused_nodes", "benchmark_codegened_module"):

                def spy(scheduler, *args, _name=name, _run=getattr(Scheduler, name)):
                    events.append(_name)
                    return _run(scheduler, *args)

                stack.enter_context(unittest.mock.patch.object(Scheduler, name, spy))
            return stack

        def accept_then_defer(self, fn, args, accept_epilogue):
            """Compiles ``fn`` with a hook that accepts the side ``accept_epilogue``
            names, and defers (None) the other side once the buffer is finalized.
            The scheduler offers the epilogue first, so before that the other
            side is declined, to be offered again in the next fusion round.
            Returns the accepted call, the first deferred call, the names of
            what was benchmarked after it, and a summary of every call."""
            events = []

            def decide(call):
                events.append(call)
                if call.epilogue == accept_epilogue:
                    return _accept()(call)
                if call.finalized is None:
                    return MultiTemplateFusionDecision(fuse=False)
                return None

            choices = RecordingFusionChoices(decide)
            with self._benchmarks(events):
                self.compile(fn, *args, choices=choices)
            calls = choices.calls
            seen = [
                (c.epilogue, c.finalized is not None, c.marked_unsafe, c.decision)
                for c in calls
            ]
            accepted = next((c for c in calls if c.epilogue == accept_epilogue), None)
            self.assertIsNotNone(accepted, seen)
            deferred = next(
                (
                    c
                    for c in calls[calls.index(accepted) + 1 :]
                    if c.epilogue != accept_epilogue
                ),
                None,
            )
            self.assertIsNotNone(deferred, seen)
            self.assertIsNone(deferred.decision, seen)
            after = events[next(i for i, e in enumerate(events) if e is deferred) + 1 :]
            return accepted, deferred, [e for e in after if isinstance(e, str)], seen

        @staticmethod
        def _gather_mm_relu(source, indices, y):
            return (source[indices + 1] @ y).relu()

        @staticmethod
        def _upcast_mm_mul_gather(x, y, source, indices):
            return (x.to(y.dtype) @ y) * source[indices + 1]

        def _gather_args(self):
            # A computed index: unlike an index input, which the autotune block
            # fills with zeros, it can be out of range there.
            source = torch.randn(512, 256, device=GPU_TYPE)
            return source, torch.randint(0, 511, (256,), device=GPU_TYPE)

        @fresh_cache()
        def test_an_accepted_producer_with_an_assert_is_not_benchmarked_later(self):
            # The gather's bounds check could fail on generated inputs, so the
            # deferred epilogue is not benchmarked with it.
            source, indices = self._gather_args()
            y = torch.randn(256, 256, device=GPU_TYPE)
            accepted, deferred, benchmarked, seen = self.accept_then_defer(
                self._gather_mm_relu, (source, indices, y), accept_epilogue=False
            )
            self.assertFalse(accepted.marked_unsafe, seen)
            self.assertTrue(deferred.marked_unsafe, seen)
            self.assertEqual(benchmarked, [], (benchmarked, seen))

            # Control: an upcast producer has no assert; the epilogue is
            # benchmarked.
            accepted, deferred, benchmarked, seen = self.accept_then_defer(
                self._upcast_mm_relu, self._upcast_args(), accept_epilogue=False
            )
            self.assertFalse(deferred.marked_unsafe, seen)
            self.assertTrue(benchmarked, (benchmarked, seen))

        @fresh_cache()
        def test_an_accepted_epilogue_with_an_assert_is_not_benchmarked_later(self):
            x, y = self._upcast_args()
            accepted, deferred, benchmarked, seen = self.accept_then_defer(
                self._upcast_mm_mul_gather,
                (x, y, *self._gather_args()),
                accept_epilogue=True,
            )
            self.assertFalse(accepted.marked_unsafe, seen)
            self.assertTrue(deferred.marked_unsafe, seen)
            self.assertEqual(benchmarked, [], (benchmarked, seen))

            # Control: a relu epilogue has no assert; the producer is benchmarked.
            accepted, deferred, benchmarked, seen = self.accept_then_defer(
                self._upcast_mm_relu, self._upcast_args(), accept_epilogue=True
            )
            self.assertFalse(deferred.marked_unsafe, seen)
            self.assertTrue(benchmarked, (benchmarked, seen))

        @fresh_cache()
        @config.patch(
            {
                "multi_kernel_hints": [64],
                "max_autotune_gemm_backends": "ATEN,TRITON",
                "test_configs.max_mm_configs": None,
            }
        )
        def test_the_hook_is_not_consulted_with_multi_kernel_hints(self):
            choices = RecordingFusionChoices(_accept())
            self.compile(
                self._mm_relu,
                self._fp16(256, 256),
                self._fp16(256, 256),
                choices=choices,
            )
            self.assertEqual(choices.calls, [])

        @fresh_cache()
        def test_a_decision_without_a_valid_template_is_declined(self):
            args = (self._fp16(256, 256), self._fp16(256, 256))

            def extern(call):
                choice = next(
                    c
                    for c in call.multi_node.choices
                    if isinstance(c, ExternKernelCaller)
                )
                return MultiTemplateFusionDecision(fuse=True, choice=choice)

            for name, decide, backends in (
                (
                    "no choice for an unfinalized buffer",
                    lambda call: MultiTemplateFusionDecision(fuse=True),
                    "TRITON",
                ),
                ("an extern choice", extern, "ATEN,TRITON"),
            ):
                with (
                    self.subTest(name),
                    config.patch(max_autotune_gemm_backends=backends),
                ):
                    choices = RecordingFusionChoices(decide)
                    code = self.compile(self._mm_relu, *args, choices=choices)
                    self.assertTrue(choices.calls)
                    # Unfused, the buffer takes its fastest choice, which may be
                    # the extern kernel.
                    self.assertNotIn("triton_tem_fused_mm_relu", code)
                    FileCheck().check("triton_poi_fused_relu").run(code)

        @fresh_cache()
        def test_a_template_of_another_buffer_is_declined(self):
            def two_mm_relu(a, b, c, d):
                return (a @ b).relu(), (c @ d).relu()

            first_buffer = []

            def decide(call):
                # The first buffer fuses with its own template; the other is
                # offered that same caller.
                if not first_buffer:
                    first_buffer.append(call.multi_node)
                    return _accept()(call)
                return MultiTemplateFusionDecision(
                    fuse=True, choice=_triton_choices(first_buffer[0])[0]
                )

            choices = RecordingFusionChoices(decide)
            code = self.compile(
                two_mm_relu, *(self._fp16(256, 256) for _ in range(4)), choices=choices
            )
            self.assertTrue(
                any(c.multi_node is not first_buffer[0] for c in choices.calls)
            )
            FileCheck().check_count(
                "def triton_tem_fused_mm_relu", 1, exactly=True
            ).check_count("def triton_poi_fused_relu", 1, exactly=True).run(code)

        @unittest.skipIf(
            not has_triton_cuda_tma_device(), "Needs TMA-load template choices"
        )
        @fresh_cache()
        @config.patch(
            {
                "triton.enable_persistent_tma_matmul": True,
                "test_configs.max_mm_configs": None,
            }
        )
        def test_a_template_that_cannot_take_the_producer_is_declined(self):
            # Persistent TMA templates load their inputs with TMA, which a
            # producer cannot be fused into.
            def decide(call):
                inputs = {i.get_name() for i in call.multi_node.inputs}
                fed = inputs & set(call.other_buffers)
                unsupported = [
                    c
                    for c in _triton_choices(call.multi_node)
                    if not fed <= c.load_input_fusion_allowed_inputs
                ]
                return MultiTemplateFusionDecision(fuse=True, choice=unsupported[0])

            choices = RecordingFusionChoices(decide)
            code = self.compile(self._upcast_mm, *self._upcast_args(), choices=choices)
            self.assertTrue(choices.calls)
            self.assertFalse(any(c.epilogue for c in choices.calls))
            self.assert_kernels(code, 2)

        @unittest.skipIf(
            not has_triton_cuda_tma_device(), "Needs TMA-store template choices"
        )
        @fresh_cache()
        @config.patch(
            {
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_template_tma_store": True,
                "test_configs.max_mm_configs": None,
            }
        )
        def test_a_fused_kernel_that_does_not_compile_is_declined(self):
            # The TMA-store/view repro: a TMA-store template cannot split this
            # epilogue (CantSplit), so choosing one declines the fusion.
            def f(a, b, bias):
                return (a @ b).view(256, 3, 256) * bias.view(3, 1)

            orig_generate = TritonTemplate.generate

            def generate(self, *args, **kwargs):
                choice = orig_generate(self, *args, **kwargs)
                if choice is not None and kwargs.get("tma_store", False):
                    choice._test_tma_store = True
                return choice

            def decide(call):
                [choice, *_] = [
                    c
                    for c in _triton_choices(call.multi_node)
                    if getattr(c, "_test_tma_store", False)
                ]
                return MultiTemplateFusionDecision(fuse=True, choice=choice)

            choices = RecordingFusionChoices(decide)
            args = (
                self._fp16(384, 256),
                self._fp16(256, 512),
                self._fp16(3),
            )
            with unittest.mock.patch.object(TritonTemplate, "generate", generate):
                code = self.compile(f, *args, choices=choices)
            self.assertTrue(choices.calls)
            self.assert_kernels(code, 2)


if HAS_CPU and not torch.backends.mps.is_available():

    class BenchmarkFusionCpuTest(TestCase):
        common = check_model
        device = "cpu"

    copy_tests(BenchmarkFusionTestTemplate, BenchmarkFusionCpuTest, "cpu")

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU_AND_TRITON:
        run_tests()
