# Owner(s): ["module: inductor"]
"""Integration tests for the optional ``experimental_cpp_ir`` Inductor path.

Scope (this file only; no core edits):
- Flag defaults off and leaves the stock route alone; the portable config
  snapshot separates FX-cache entries by flag value.
- With the flag on, ``_InProcessFxCompile.codegen_and_compile`` still runs
  the normal ``_recursive_post_grad_passes`` and delegates to
  ``compile_fx_loops_ir_output`` instead of constructing ``GraphLowering``.
- The existing ``loops_ir_backend`` callable is a thin ``compile_fx``
  forwarder selecting the flag.
- End-to-end bitwise equality with original eager (finite values, signed
  zeros, NaN payloads, subnormals, infinities, FMA-sensitive chains),
  views with nonzero offsets, repeated calls preserving prior outputs, and
  lifted parameters with ``requires_grad=True`` in inference.
- The direct helper returns the normal ``CompiledFxGraph`` with file-backed
  source; the emitted wrapper uses normal ``AsyncCompile.triton`` /
  ``CachingAutotuner`` fixed-grid launches with no C++ IR interpretation in
  the hot body.
- FX-cache hits reload without C++ recompilation; explicit rejections for
  dynamic shapes, backward, input/storage mutation, and unsupported wrapper
  modes; cudagraphs disabled with an accurate reason; the compiler
  fingerprint covers the native C++ IR sources (isolated-filesystem mock,
  never editing real sources).

Environment: unit kernels only (tiny tensors), ``CUDA_VISIBLE_DEVICES=5``,
``CC=/usr/bin/gcc CXX=/usr/bin/g++ MAX_JOBS=2``. No benchmark timings.
"""

import io
import os
import unittest
from unittest import mock

import torch

from torch._inductor.test_case import run_tests, TestCase


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required")


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    return (
        tensor.detach()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    )


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.shape != expected.shape:
        raise AssertionError((actual.shape, expected.shape))
    if actual.stride() != expected.stride():
        raise AssertionError((actual.stride(), expected.stride()))
    if actual.dtype != expected.dtype:
        raise AssertionError((actual.dtype, expected.dtype))
    a, b = _raw_bytes(actual), _raw_bytes(expected)
    if a != b:
        raise AssertionError(
            f"byte mismatch: len {len(a)} vs {len(b)}; first differing byte at "
            f"{next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)}"
        )


def _cpp_ir_options(**overrides):
    opts = {"experimental_cpp_ir": True, "triton.cudagraphs": False}
    opts.update(overrides)
    return opts


def _compile_with_cpp_ir(fn, *inputs, dynamic=False, **opt_overrides):
    return torch.compile(
        fn,
        backend="inductor",
        options=_cpp_ir_options(**opt_overrides),
        fullgraph=True,
        dynamic=dynamic,
    )(*inputs)


class TestLoopsIrIntegration(TestCase):
    # ------------------------------------------------------------------
    # Flag plumbing (CPU-runnable).
    # ------------------------------------------------------------------

    def test_flag_defaults_off(self):
        from torch._inductor import config as inductor_config

        if os.environ.get("TORCHINDUCTOR_EXPERIMENTAL_CPP_IR", "0") == "1":
            self.skipTest("flag forced on via environment")
        self.assertFalse(inductor_config.experimental_cpp_ir)

    def test_flag_in_portable_config_snapshot(self):
        from torch._inductor import config as inductor_config

        base = inductor_config.save_config_portable()
        self.assertIn("experimental_cpp_ir", base)
        self.assertFalse(base["experimental_cpp_ir"])
        with inductor_config.patch({"experimental_cpp_ir": True}):
            patched = inductor_config.save_config_portable()
        self.assertTrue(patched["experimental_cpp_ir"])
        self.assertNotEqual(base["experimental_cpp_ir"], patched["experimental_cpp_ir"])

    def test_pooling_decomposition_requires_stock_lowering(self):
        from torch._inductor import config as inductor_config
        from torch._inductor.decomposition import select_decomp_table

        operators = (
            torch.ops.aten.max_pool2d_with_indices.default,
            torch.ops.aten.max_pool3d_with_indices.default,
        )
        with inductor_config.patch({"experimental_cpp_ir": False}):
            stock = select_decomp_table()
            for operator in operators:
                self.assertIn(operator, stock)
        for numerics in ("strict_eager", "inductor"):
            with self.subTest(numerics=numerics):
                with inductor_config.patch(
                    {
                        "experimental_cpp_ir": True,
                        "experimental_cpp_ir_numerics": numerics,
                    }
                ):
                    table = select_decomp_table()
                    for operator in operators:
                        self.assertNotIn(operator, table)

    def test_native_pooling_matches_eager_bits(self):
        _require_cuda()

        def function(value):
            return torch.nn.functional.max_pool2d(value, 3, stride=2, padding=1)

        value = torch.randn(2, 3, 17, 17, device="cuda")
        with torch.inference_mode():
            actual = _compile_with_cpp_ir(function, value)
            expected = function(value)
        self.assertEqual(actual.view(torch.uint8), expected.view(torch.uint8))

    def test_numerical_policy_is_portable_and_strict_by_default(self):
        from torch._inductor import config as inductor_config
        from torch._inductor.fx_passes.cpp_ir_strict import cpp_ir_strict_enabled

        base = inductor_config.save_config_portable()
        self.assertEqual(base["experimental_cpp_ir_numerics"], "strict_eager")
        with inductor_config.patch({"experimental_cpp_ir": True}):
            self.assertTrue(cpp_ir_strict_enabled())
        with inductor_config.patch(
            {
                "experimental_cpp_ir": True,
                "experimental_cpp_ir_numerics": "inductor",
            }
        ):
            self.assertFalse(cpp_ir_strict_enabled())
            snapshot = inductor_config.save_config_portable()
            self.assertEqual(snapshot["experimental_cpp_ir_numerics"], "inductor")

    def test_typed_extern_metadata_freezes_and_renders(self):
        from types import SimpleNamespace

        from torch._inductor.codegen.loops_ir_wrapper import render_frozen_arg
        from torch._inductor.loops_ir_backend import _freeze_call_arg

        for kind, text, expected, expression in (
            ("layout", "strided", torch.strided, "torch.strided"),
            (
                "memory_format",
                "contiguous_format",
                torch.contiguous_format,
                "torch.contiguous_format",
            ),
            (
                "memory_format",
                "preserve_format",
                torch.preserve_format,
                "torch.preserve_format",
            ),
        ):
            with self.subTest(kind=kind, text=text):
                frozen = _freeze_call_arg(SimpleNamespace(kind_name=kind, text=text))
                self.assertIs(frozen, expected)
                self.assertEqual(render_frozen_arg(frozen, {}), expression)
        with self.assertRaisesRegex(Exception, "unsupported|invalid"):
            _freeze_call_arg(SimpleNamespace(kind_name="layout", text="arbitrary_code"))

    def test_full_layout_and_clone_memory_format_execute(self):
        _require_cuda()
        self.addCleanup(torch._dynamo.reset)

        def function(value):
            filled = torch.full(
                value.shape,
                0.25,
                dtype=value.dtype,
                layout=torch.strided,
                device=value.device,
            )
            cloned = filled.clone(memory_format=torch.contiguous_format)
            return value * cloned + filled

        value = torch.randn(4, 16, device="cuda")
        for numerics in ("strict_eager", "inductor"):
            with self.subTest(numerics=numerics), torch.inference_mode():
                actual = _compile_with_cpp_ir(
                    function, value, experimental_cpp_ir_numerics=numerics
                )
                expected = function(value)
                _assert_bitwise_equal(actual, expected)

    def test_non_float_boundary_views_preserve_aliases(self):
        _require_cuda()
        self.addCleanup(torch._dynamo.reset)
        indices = torch.arange(16, device="cuda", dtype=torch.int64)[1:7:2].view(1, 3)
        mask = torch.tensor([[True], [False], [True]], device="cuda")
        value = torch.randn(4, 3, device="cuda")

        def function(index, keep, data):
            return index.expand(4, 3), keep.transpose(0, 1).expand(4, 3), data + 1.0

        for numerics in ("strict_eager", "inductor"):
            with self.subTest(numerics=numerics), torch.inference_mode():
                actual = _compile_with_cpp_ir(
                    function,
                    indices,
                    mask,
                    value,
                    experimental_cpp_ir_numerics=numerics,
                )
                expected = function(indices, mask, value)
                for result, reference in zip(actual, expected):
                    _assert_bitwise_equal(result, reference)
                self.assertEqual(
                    actual[0].untyped_storage().data_ptr(),
                    indices.untyped_storage().data_ptr(),
                )
                self.assertEqual(
                    actual[1].untyped_storage().data_ptr(),
                    mask.untyped_storage().data_ptr(),
                )

    def test_bert_normalization_and_native_math_preserve_strict_bits(self):
        _require_cuda()
        self.addCleanup(torch._dynamo.reset)
        value = torch.randn(8, 32, device="cuda")
        weight = torch.randn(32, device="cuda")
        bias = torch.randn(32, device="cuda")

        def function(data, scale, offset):
            mean = data.mean(-1, keepdim=True)
            std = data.std(-1, keepdim=True)
            normalized = scale * (data - mean) / (std + 1e-6) + offset
            return (
                normalized,
                torch.nn.functional.gelu(data),
                torch.nn.functional.softmax(data, dim=-1),
            )

        with torch.inference_mode():
            actual = _compile_with_cpp_ir(function, value, weight, bias)
            expected = function(value, weight, bias)
        for result, reference in zip(actual, expected):
            _assert_bitwise_equal(result, reference)

    def test_numerical_modes_do_not_change_ambient_precision(self):
        _require_cuda()
        self.addCleanup(torch._dynamo.reset)

        def settings():
            return (
                torch.get_float32_matmul_precision(),
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
                torch.are_deterministic_algorithms_enabled(),
            )

        def function(value):
            return torch.relu(value * 1.25 + 0.5)

        value = torch.randn(32, device="cuda")
        before = settings()
        for numerics in ("strict_eager", "inductor"):
            with self.subTest(numerics=numerics), torch.inference_mode():
                actual = _compile_with_cpp_ir(
                    function, value, experimental_cpp_ir_numerics=numerics
                )
                if numerics == "strict_eager":
                    _assert_bitwise_equal(actual, function(value))
                else:
                    torch.testing.assert_close(actual, function(value))
                self.assertEqual(settings(), before)

    def test_backend_forwards_to_compile_fx_with_flag(self):
        from torch._inductor.loops_ir_backend import loops_ir_backend
        from torch.fx.experimental.proxy_tensor import make_fx

        x = torch.randn(4)
        gm = make_fx(lambda a: a * 2.0 + 1.0)(x)
        with mock.patch(
            "torch._inductor.compile_fx.compile_fx", return_value="FORWARDED"
        ) as inner:
            out = loops_ir_backend(gm, [x])
        self.assertEqual(out, "FORWARDED")
        self.assertEqual(
            inner.call_args.kwargs.get("config_patches"),
            {"experimental_cpp_ir": True},
        )

    # ------------------------------------------------------------------
    # Lowering-seam routing (CPU-runnable; everything past the seam is
    # mocked, so no GPU or C++ build is needed).
    # ------------------------------------------------------------------

    def _make_seam_graph(self):
        from torch.fx.experimental.proxy_tensor import make_fx

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        gm = make_fx(lambda a, b: torch.relu(a * b + b))(x, y)
        gm.shape_env = None
        return gm, [x, y]

    def _seam_kwargs(self, **overrides):
        from torch._inductor.utils import BoxedBool

        kwargs = {
            "cudagraphs": BoxedBool(False),
            "static_input_idxs": (),
            "is_backward": False,
            "is_inference": True,
            "cpp_wrapper": False,
            "aot_mode": False,
            "fx_wrapper": False,
            "extern_node_serializer": None,
        }
        kwargs.update(overrides)
        return kwargs

    def _null_virtualized(self):
        fake_v = mock.MagicMock(name="virtualized")
        fake_v.aot_compilation = False
        return fake_v

    def test_flag_on_runs_post_grad_and_skips_graph_lowering(self):
        import torch._inductor.compile_fx as compile_fx_mod
        from torch._inductor import config as inductor_config

        gm, inputs = self._make_seam_graph()
        kwargs = self._seam_kwargs()
        with (
            mock.patch.object(
                compile_fx_mod,
                "_recursive_post_grad_passes",
                wraps=compile_fx_mod._recursive_post_grad_passes,
            ) as post_grad,
            mock.patch.object(
                compile_fx_mod,
                "GraphLowering",
                side_effect=AssertionError("GraphLowering must not be constructed"),
            ) as graph_lowering,
            mock.patch(
                "torch._inductor.loops_ir_backend.compile_fx_loops_ir_output",
                return_value="SEAM_SENTINEL",
            ) as helper,
            mock.patch.object(compile_fx_mod, "V", self._null_virtualized()),
            inductor_config.patch({"experimental_cpp_ir": True}),
        ):
            out = compile_fx_mod._InProcessFxCompile().codegen_and_compile(
                gm, inputs, [], kwargs
            )
        self.assertEqual(out, "SEAM_SENTINEL")
        self.assertTrue(post_grad.called, "post-grad passes must still run")
        graph_lowering.assert_not_called()
        helper.assert_called_once()
        call = helper.call_args
        self.assertIs(call.kwargs["graph_kwargs"], kwargs)
        self.assertEqual(call.kwargs["inputs_to_check"], [])
        self.assertIsInstance(call.kwargs["runnable_graph_str"], str)
        self.assertIsInstance(call.kwargs["inductor_post_grad_graph_str"], str)

    def test_flag_off_uses_graph_lowering_not_cpp_helper(self):
        import torch._inductor.compile_fx as compile_fx_mod
        from torch._inductor import config as inductor_config

        gm, inputs = self._make_seam_graph()
        kwargs = self._seam_kwargs()
        with (
            mock.patch.object(
                compile_fx_mod,
                "GraphLowering",
                side_effect=RuntimeError("STOCK_ROUTE"),
            ) as graph_lowering,
            mock.patch(
                "torch._inductor.loops_ir_backend.compile_fx_loops_ir_output",
                side_effect=AssertionError("cpp helper must not run on stock path"),
            ) as helper,
            mock.patch.object(compile_fx_mod, "V", self._null_virtualized()),
            inductor_config.patch({"experimental_cpp_ir": False}),
        ):
            with self.assertRaisesRegex(RuntimeError, "STOCK_ROUTE"):
                compile_fx_mod._InProcessFxCompile().codegen_and_compile(
                    gm, inputs, [], kwargs
                )
        graph_lowering.assert_called()
        helper.assert_not_called()

    # ------------------------------------------------------------------
    # Explicit helper rejections (CPU-runnable; raised before any device
    # checks or C++ work).
    # ------------------------------------------------------------------

    def test_helper_rejects_unsupported_modes(self):
        from torch._inductor.loops_ir import LoopsIrUnsupportedError
        from torch._inductor.loops_ir_backend import compile_fx_loops_ir_output
        from torch.fx.experimental.proxy_tensor import make_fx

        x = torch.randn(8)
        gm = make_fx(lambda a: a * 2.0 + 1.0)(x)
        cases = [
            ("backward", {"is_backward": True}, "backward"),
            ("cpp_wrapper", {"cpp_wrapper": True}, "wrapper"),
            ("aot_mode", {"aot_mode": True}, "wrapper"),
            ("fx_wrapper", {"fx_wrapper": True}, "wrapper"),
            (
                "extern_serializer",
                {"extern_node_serializer": object()},
                "extern_node_serializer",
            ),
        ]
        for name, override, pattern in cases:
            with self.subTest(mode=name):
                with self.assertRaisesRegex(LoopsIrUnsupportedError, pattern):
                    compile_fx_loops_ir_output(
                        gm,
                        [x],
                        graph_kwargs=self._seam_kwargs(**override),
                        inputs_to_check=(),
                        runnable_graph_str="",
                        inductor_post_grad_graph_str="",
                    )

    def test_helper_rejects_non_cuda_inputs(self):
        from torch._inductor.loops_ir import LoopsIrUnsupportedError
        from torch._inductor.loops_ir_backend import compile_fx_loops_ir_output
        from torch.fx.experimental.proxy_tensor import make_fx

        x = torch.randn(8)
        gm = make_fx(lambda a: a * 2.0 + 1.0)(x)
        with self.assertRaisesRegex(LoopsIrUnsupportedError, "CUDA"):
            compile_fx_loops_ir_output(
                gm,
                [x],
                graph_kwargs=self._seam_kwargs(),
                inputs_to_check=(),
                runnable_graph_str="",
                inductor_post_grad_graph_str="",
            )

    # ------------------------------------------------------------------
    # Compiler fingerprint covers the native C++ IR sources (CPU-runnable;
    # isolated-filesystem mock, never touches real sources).
    # ------------------------------------------------------------------

    def test_fingerprint_covers_cpp_sources(self):
        from torch._inductor import codecache
        from torch._inductor.codecache import torch_key

        self.assertTrue(hasattr(torch_key, "clear"))
        inductor_root = os.path.dirname(codecache.__file__)
        cpp_ir_root = os.path.normpath(
            os.path.join(inductor_root, "../csrc/inductor/experimental/loops_ir")
        )
        self.assertTrue(
            os.path.isdir(cpp_ir_root), f"expected C++ IR dir at {cpp_ir_root}"
        )
        blobs = {
            "alpha.cpp": b"int alpha() { return 1; }",
            "beta.h": b"#pragma once",
            "notes.py": b"x = 1",
        }
        real_isdir = os.path.isdir
        real_scandir = os.scandir
        real_open = open

        class _Entry:
            def __init__(self, name):
                self.name = name

            def is_file(self):
                return True

        class _Scan:
            def __init__(self, names):
                self._names = names

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                return iter(_Entry(name) for name in self._names)

        def fake_isdir(path, *args, **kwargs):
            if os.path.normpath(str(path)) == cpp_ir_root:
                return True
            return real_isdir(path, *args, **kwargs)

        def fake_scandir(path, *args, **kwargs):
            if os.path.normpath(str(path)) == cpp_ir_root:
                return _Scan(sorted(blobs))
            return real_scandir(path, *args, **kwargs)

        def fake_open(path, *args, **kwargs):
            if isinstance(path, (str, os.PathLike)) and os.path.normpath(
                str(path)
            ).startswith(cpp_ir_root):
                return io.BytesIO(blobs[os.path.basename(str(path))])
            return real_open(path, *args, **kwargs)

        try:
            torch_key.clear()
            with (
                mock.patch("os.path.isdir", side_effect=fake_isdir),
                mock.patch("os.scandir", side_effect=fake_scandir),
                mock.patch("builtins.open", side_effect=fake_open),
            ):
                key_base = torch_key()
                torch_key.clear()
                blobs["alpha.cpp"] = b"int alpha() { return 2; }"
                key_cpp_changed = torch_key()
                torch_key.clear()
                blobs["alpha.cpp"] = b"int alpha() { return 1; }"
                blobs["notes.py"] = b"x = 2  # not a native source"
                key_py_changed = torch_key()
        finally:
            torch_key.clear()
        self.assertNotEqual(
            key_base, key_cpp_changed, "native .cpp changes must invalidate torch_key"
        )
        self.assertEqual(
            key_base, key_py_changed, "non-native files in the dir must not affect it"
        )

    # ------------------------------------------------------------------
    # Eager-exact execution through the integrated option (CUDA).
    # ------------------------------------------------------------------

    def test_option_pointwise_chain_matches_eager_bits(self):
        _require_cuda()
        self.addCleanup(torch._dynamo.reset)
        torch.manual_seed(0)
        x = torch.randn(16, 16, device="cuda", dtype=torch.float32)
        y = torch.randn(16, 16, device="cuda", dtype=torch.float32)
        fn = lambda a, b: torch.relu(a * b + b)  # noqa: E731
        with torch.inference_mode():
            expected = fn(x, y)
            actual = _compile_with_cpp_ir(fn, x, y)
        _assert_bitwise_equal(actual, expected)

    def test_option_special_values_match_eager_bits(self):
        _require_cuda()
        self.addCleanup(torch._dynamo.reset)
        vals = torch.tensor(
            [
                0.0,
                -0.0,
                1.0,
                -1.0,
                float("inf"),
                float("-inf"),
                1.1754944e-38,  # min normal
                1.4e-45,  # min subnormal
                1e-30,
                -1e-30,
                3.4028235e38,
                -3.4028235e38,
            ],
            device="cuda",
            dtype=torch.float32,
        )
        nan_payload = torch.tensor(
            [0x7FC00001, 0x7FC00002, 0x7F800001], dtype=torch.int32, device="cuda"
        ).view(torch.float32)
        x = torch.cat([vals, nan_payload])
        for name, fn in (
            ("mul_one", lambda v: v * 1.0),
            ("add_neg_zero", lambda v: v + (-0.0)),
            ("add_pos_zero", lambda v: v + 0.0),
            ("relu", lambda v: torch.relu(v)),
            ("chain", lambda v: torch.relu(v * 1.5 + 0.25)),
        ):
            with self.subTest(operation=name):
                torch._dynamo.reset()
                with torch.inference_mode():
                    _assert_bitwise_equal(_compile_with_cpp_ir(fn, x), fn(x))

    def test_option_fma_sensitive_chain_matches_eager_bits(self):
        _require_cuda()
        self.addCleanup(torch._dynamo.reset)
        torch.manual_seed(7)
        a = torch.randn(256, device="cuda", dtype=torch.float32)
        b = torch.randn(256, device="cuda", dtype=torch.float32)
        c = torch.randn(256, device="cuda", dtype=torch.float32)
        fn = lambda x, y, z: (x * y) + z  # noqa: E731
        with torch.inference_mode():
            _assert_bitwise_equal(_compile_with_cpp_ir(fn, a, b, c), fn(a, b, c))

    def test_option_view_with_nonzero_offset(self):
        _require_cuda()
        self.addCleanup(torch._dynamo.reset)
        torch.manual_seed(3)
        base = torch.randn(8, 16, device="cuda", dtype=torch.float32)
        x = base[1:7, ::2]  # non-contiguous slice, nonzero storage offset
        self.assertGreater(x.storage_offset(), 0)
        w = torch.randn(6, 8, device="cuda", dtype=torch.float32)
        fn = lambda a, b: torch.relu(a * b + 1.0)  # noqa: E731
        with torch.inference_mode():
            expected = fn(x, w)
            actual = _compile_with_cpp_ir(fn, x, w)
        self.assertEqual(tuple(actual.stride()), tuple(expected.stride()))
        _assert_bitwise_equal(actual, expected)

    def test_option_repeated_calls_preserve_prior_outputs(self):
        _require_cuda()
        self.addCleanup(torch._dynamo.reset)
        torch.manual_seed(11)
        fn = lambda a: torch.relu(a * 2.0 + 1.0)  # noqa: E731
        compiled = torch.compile(
            fn,
            backend="inductor",
            options=_cpp_ir_options(),
            fullgraph=True,
            dynamic=False,
        )
        inputs = [torch.randn(32, device="cuda", dtype=torch.float32) for _ in range(2)]
        with torch.inference_mode():
            outputs = [compiled(value) for value in inputs]
            snapshot = outputs[0].clone()
            for actual, value in zip(outputs, inputs):
                _assert_bitwise_equal(actual, fn(value))
        _assert_bitwise_equal(outputs[0], snapshot)
        self.assertNotEqual(outputs[0].data_ptr(), outputs[1].data_ptr())

    def test_option_lifted_parameter_requires_grad_in_inference(self):
        _require_cuda()
        self.addCleanup(torch._dynamo.reset)
        torch.manual_seed(5)
        lin = torch.nn.Linear(16, 16, device="cuda", dtype=torch.float32)
        self.assertTrue(any(p.requires_grad for p in lin.parameters()))
        x = torch.randn(8, 16, device="cuda", dtype=torch.float32)
        fn = lambda t: torch.relu(lin(t))  # noqa: E731
        with torch.inference_mode():
            _assert_bitwise_equal(_compile_with_cpp_ir(fn, x), fn(x))

    # ------------------------------------------------------------------
    # Direct helper: normal CompiledFxGraph + normal launch machinery.
    # ------------------------------------------------------------------

    def test_direct_helper_returns_file_backed_compiled_fx_graph(self):
        _require_cuda()
        from torch._inductor.loops_ir_backend import compile_fx_loops_ir_output
        from torch._inductor.output_code import CompiledFxGraph
        from torch.fx.experimental.proxy_tensor import make_fx

        torch.manual_seed(0)
        x = torch.randn(32, 32, device="cuda", dtype=torch.float32)
        y = torch.randn(32, 32, device="cuda", dtype=torch.float32)
        fn = lambda a, b: torch.relu(a * b + b)  # noqa: E731
        gm = make_fx(fn)(x, y)
        graph = compile_fx_loops_ir_output(
            gm,
            [x, y],
            graph_kwargs=self._seam_kwargs(),
            inputs_to_check=(),
            runnable_graph_str="runnable",
            inductor_post_grad_graph_str="post_grad",
        )
        self.assertIsInstance(graph, CompiledFxGraph)
        self.assertTrue(graph._boxed_call)
        self.assertTrue(graph.cache_key)
        self.assertIn("def call(", graph.source_code)
        self.assertIn("cuda", set(graph.device_types))
        self.assertEqual(set(graph.mutated_inputs), set())
        flat = graph.current_callable([x, y])
        self.assertIsInstance(flat, tuple)
        with torch.inference_mode():
            _assert_bitwise_equal(flat[0], fn(x, y))

    def test_wrapper_uses_fixed_config_cached_autotuner(self):
        _require_cuda()
        from torch._inductor.loops_ir_backend import compile_fx_entry
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        from torch.fx.experimental.proxy_tensor import make_fx

        torch.manual_seed(0)
        x = torch.randn(64, device="cuda", dtype=torch.float32)
        y = torch.randn(64, device="cuda", dtype=torch.float32)
        run = compile_fx_entry(
            make_fx(lambda a, b: torch.relu(a * b + b))(x, y), [x, y]
        )
        mod = getattr(run, "_loops_ir_module", None)
        self.assertIsNotNone(mod, "direct helper must expose its wrapper module")
        with open(mod.__file__) as handle:
            source = handle.read()
        for snippet in (
            "async_compile.triton(",
            "async_compile.wait(globals())",
            "def call(args):",
            ".run(",
            "stream=",
        ):
            self.assertIn(snippet, source)
        for snippet in (
            "kind_name",
            "CallArg",
            "fused.nodes",
            "_materialize_call_arg",
            "_TensorKey",
        ):
            self.assertNotIn(snippet, source)
        sources = getattr(run, "_loops_ir_sources", None)
        self.assertTrue(sources)
        self.assertTrue(
            any("fixed_config(" in text for text in sources.values()),
            "generated kernels must use the normal fixed-config path",
        )
        tuners = [
            value for value in vars(mod).values() if isinstance(value, CachingAutotuner)
        ]
        self.assertGreater(len(tuners), 0, "expected CachingAutotuner launchers")
        with torch.inference_mode():
            run([x, y])
        counts = [len(tuner.launchers) for tuner in tuners]
        self.assertTrue(any(count >= 1 for count in counts))
        with torch.inference_mode():
            run([x, y])
        self.assertEqual(
            [len(tuner.launchers) for tuner in tuners],
            counts,
            "launchers must be reused, not recompiled per call",
        )

    def test_cudagraphs_disabled_with_accurate_reason(self):
        _require_cuda()
        from torch._inductor.loops_ir_backend import compile_fx_loops_ir_output
        from torch._inductor.output_code import CompiledFxGraph
        from torch._inductor.utils import BoxedBool
        from torch.fx.experimental.proxy_tensor import make_fx

        x = torch.randn(16, device="cuda", dtype=torch.float32)
        gm = make_fx(lambda a: torch.relu(a * 2.0 + 1.0))(x)
        graph = compile_fx_loops_ir_output(
            gm,
            [x],
            graph_kwargs=self._seam_kwargs(cudagraphs=BoxedBool(True)),
            inputs_to_check=(),
            runnable_graph_str="",
            inductor_post_grad_graph_str="",
        )
        self.assertIsInstance(graph, CompiledFxGraph)
        self.assertIsNotNone(graph.disabled_cudagraphs_reason)
        self.assertFalse(graph.kernel_free_cudagraph)

    # ------------------------------------------------------------------
    # Cache behavior + explicit dynamic/mutation rejection (CUDA).
    # ------------------------------------------------------------------

    def test_fx_cache_hit_avoids_cpp_recompile_and_flag_separation(self):
        _require_cuda()
        from torch._inductor import loops_ir_backend as backend_mod
        from torch._inductor.utils import fresh_cache

        self.addCleanup(torch._dynamo.reset)
        torch.manual_seed(0)
        fn = lambda a, b: torch.relu(a * b + b)  # noqa: E731
        x = torch.randn(16, 16, device="cuda", dtype=torch.float32)
        y = torch.randn(16, 16, device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            expected = fn(x, y)
        with fresh_cache():
            torch._dynamo.reset()
            with mock.patch.object(
                backend_mod,
                "_build_frozen_plan",
                wraps=backend_mod._build_frozen_plan,
            ) as plan:
                with torch.inference_mode():
                    first = _compile_with_cpp_ir(fn, x, y)
                self.assertEqual(plan.call_count, 1)
                _assert_bitwise_equal(first, expected)
                # Same flag again: FX-cache hit must not rebuild the C++ plan.
                torch._dynamo.reset()
                with torch.inference_mode():
                    second = _compile_with_cpp_ir(fn, x, y)
                self.assertEqual(plan.call_count, 1)
                _assert_bitwise_equal(second, expected)
                # Stock flag value: separate entry, and the C++ plan builder
                # must stay untouched on the normal path.
                torch._dynamo.reset()
                with torch.inference_mode():
                    torch.compile(
                        fn,
                        backend="inductor",
                        options={"experimental_cpp_ir": False},
                        fullgraph=True,
                        dynamic=False,
                    )(x, y)
                self.assertEqual(plan.call_count, 1)
                # Back to the flag: still a hit on the original entry.
                torch._dynamo.reset()
                with torch.inference_mode():
                    third = _compile_with_cpp_ir(fn, x, y)
                self.assertEqual(plan.call_count, 1)
                _assert_bitwise_equal(third, expected)

    def test_fx_cache_separates_numerical_modes(self):
        _require_cuda()
        from torch._inductor import loops_ir_backend as backend_mod
        from torch._inductor.utils import fresh_cache

        self.addCleanup(torch._dynamo.reset)

        def function(value):
            return torch.relu(value * 1.25 + 0.5)

        value = torch.randn(16, 16, device="cuda")
        with (
            fresh_cache(),
            mock.patch.object(
                backend_mod, "_build_frozen_plan", wraps=backend_mod._build_frozen_plan
            ) as plan,
        ):
            for numerics, expected_builds in (
                ("strict_eager", 1),
                ("inductor", 2),
                ("strict_eager", 2),
                ("inductor", 2),
            ):
                torch._dynamo.reset()
                with self.subTest(numerics=numerics), torch.inference_mode():
                    actual = _compile_with_cpp_ir(
                        function, value, experimental_cpp_ir_numerics=numerics
                    )
                    if numerics == "strict_eager":
                        _assert_bitwise_equal(actual, function(value))
                    else:
                        torch.testing.assert_close(actual, function(value))
                    self.assertEqual(plan.call_count, expected_builds)

    def test_dynamic_shapes_rejected(self):
        _require_cuda()
        self.addCleanup(torch._dynamo.reset)
        fn = lambda a: torch.relu(a * 2.0 + 1.0)  # noqa: E731
        x = torch.randn(16, device="cuda", dtype=torch.float32)
        compiled = torch.compile(
            fn,
            backend="inductor",
            options=_cpp_ir_options(),
            fullgraph=True,
            dynamic=True,
        )
        with torch.inference_mode():
            with self.assertRaisesRegex(Exception, "[Ss]ymbolic|static|SymInt"):
                compiled(x)

    def test_input_mutation_rejected(self):
        _require_cuda()
        from torch._inductor.loops_ir import LoopsIrUnsupportedError
        from torch._inductor.loops_ir_backend import compile_fx_loops_ir_output
        from torch.fx.experimental.proxy_tensor import make_fx

        def fn(x):
            x.add_(1)
            return x * 2.0

        x = torch.randn(8, device="cuda", dtype=torch.float32)
        gm = make_fx(fn)(x)
        with self.assertRaisesRegex(
            LoopsIrUnsupportedError, "[Mm]utat|unsupported|in-place|temporary"
        ):
            compile_fx_loops_ir_output(
                gm,
                [x],
                graph_kwargs=self._seam_kwargs(),
                inputs_to_check=(),
                runnable_graph_str="",
                inductor_post_grad_graph_str="",
            )


if __name__ == "__main__":
    run_tests()
