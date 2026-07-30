# Owner(s): ["module: inductor"]

import warnings

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import is_big_gpu, run_and_get_code
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


aten = torch.ops.aten
_REJECTION_PREDICATES = (
    "backend",
    "topology",
    "rank",
    "batch_one_base",
    "cat_dimension",
    "shape_stride_identity",
    "users",
    "unsupported_dtype",
    "unsupported_device",
    "dtype_device_layout",
    "bmm_compatibility",
)
_RANK_REJECTION_DETAILS = ("cat_rhs", "expand", "base")


@instantiate_parametrized_tests
class CatExpandBmmTest(TestCase):
    def _assert_rejection_counters(self, expected, expected_rank_detail=None):
        """Assert that validation recorded only the expected rejection."""

        for predicate in _REJECTION_PREDICATES:
            self.assertEqual(
                counters["inductor"][f"cat_expand_bmm_rejected_{predicate}"],
                int(predicate == expected),
            )
        for detail in _RANK_REJECTION_DETAILS:
            self.assertEqual(
                counters["inductor"][f"cat_expand_bmm_rejected_rank_{detail}"],
                int(detail == expected_rank_detail),
            )

    def _run_admission_case(
        self,
        dtype,
        backends,
        *,
        device="cpu",
        rows=2,
        reduction=8,
        columns=4,
        batch=3,
        max_autotune=False,
    ):
        """Compile one backend-admission case and capture its BMM LHS stride."""

        def fn(weights, rhs):
            lhs = torch.cat(
                [weight.expand(rhs.shape[0], -1, -1) for weight in weights], dim=1
            )
            return torch.bmm(lhs, rhs)

        weights = [
            torch.randn(1, rows, reduction, device=device, dtype=dtype)
            for _ in range(2)
        ]
        rhs = torch.randn(batch, reduction, columns, device=device, dtype=dtype)
        lhs_strides = []

        def record_bmm_lhs_stride(graph):
            bmm = next(
                iter(graph.find_nodes(op="call_function", target=aten.bmm.default))
            )
            lhs_strides.append(bmm.args[0].meta["val"].stride())

        torch._dynamo.reset()
        counters.clear()
        expected = fn(weights, rhs)
        with inductor_config.patch(
            cat_expand_bmm_rewrite=True,
            max_autotune=max_autotune,
            max_autotune_gemm_backends=backends,
            post_grad_custom_post_pass=record_bmm_lhs_stride,
        ):
            actual = torch.compile(fn, fullgraph=True)(weights, rhs)
        return actual, expected, lhs_strides

    @parametrize("cat_dim", [1, 2])
    @parametrize("outer_expand", [False, True])
    @parametrize("direct_rank3", [False, True])
    def test_rewrite(self, cat_dim, outer_expand, direct_rank3):
        def fn(weights, rhs):
            lhs = torch.cat(
                [
                    (weight if direct_rank3 else weight.unsqueeze(0)).expand(
                        rhs.shape[0], -1, -1
                    )
                    for weight in weights
                ],
                dim=cat_dim,
            )
            if outer_expand:
                lhs = lhs.expand(rhs.shape[0], lhs.shape[1], lhs.shape[2])
            return torch.bmm(lhs, rhs)

        if cat_dim == 1:
            weight_shapes = [(index % 3 + 1, 8) for index in range(15)]
        else:
            weight_shapes = [(4, 2), (4, 3), (4, 3)]
        if direct_rank3:
            weight_shapes = [(1, *shape) for shape in weight_shapes]
        weights = [
            torch.randn(shape, device="cpu", dtype=torch.float16)
            for shape in weight_shapes
        ]
        rhs = torch.randn(4, 8, 6, device="cpu", dtype=torch.float16)
        lhs_strides = []

        def record_bmm_lhs_stride(graph):
            for node in graph.find_nodes(op="call_function", target=aten.bmm.default):
                lhs_strides.append(node.args[0].meta["val"].stride())

        torch._dynamo.reset()
        counters.clear()
        expected = fn(weights, rhs)
        with inductor_config.patch(
            cat_expand_bmm_rewrite=True,
            max_autotune_gemm_backends="ATEN",
            post_grad_custom_post_pass=record_bmm_lhs_stride,
        ):
            actual = torch.compile(fn, fullgraph=True)(weights, rhs)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["cat_expand_to_batch_stride_zero"], 1)
        self.assertEqual(
            counters["inductor"]["cat_expand_outer_to_batch_stride_zero"],
            int(outer_expand),
        )
        self.assertEqual(len(lhs_strides), 1)
        self.assertEqual(lhs_strides[0][0], 0)

    def test_production_topology(self):
        def fn(weights, rhs):
            lhs = torch.cat(
                [weight.expand(rhs.shape[0], -1, -1) for weight in weights], dim=1
            )
            lhs = lhs.expand(rhs.shape[0], lhs.shape[1], lhs.shape[2])
            return torch.bmm(lhs, rhs)

        weights = [
            torch.randn(1, 10, 1890, device="cpu", dtype=torch.float16)
            for _ in range(15)
        ]
        rhs = torch.randn(2, 1890, 256, device="cpu", dtype=torch.float16)
        lhs_metadata = []

        def record_bmm_lhs_metadata(graph):
            for node in graph.find_nodes(op="call_function", target=aten.bmm.default):
                lhs = node.args[0].meta["val"]
                lhs_metadata.append((lhs.shape, lhs.stride()))

        torch._dynamo.reset()
        counters.clear()
        torch._dynamo.mark_dynamic(rhs, 0)
        compiled = torch.compile(fn, fullgraph=True)
        with inductor_config.patch(
            cat_expand_bmm_rewrite=True,
            max_autotune_gemm_backends="ATEN",
            post_grad_custom_post_pass=record_bmm_lhs_metadata,
        ):
            actual, (code,) = run_and_get_code(compiled, weights, rhs)
            second_rhs = torch.randn(3, 1890, 256, device="cpu", dtype=torch.float16)
            second_actual = compiled(weights, second_rhs)

        torch.testing.assert_close(actual, fn(weights, rhs))
        torch.testing.assert_close(second_actual, fn(weights, second_rhs))
        self.assertEqual(counters["inductor"]["cat_expand_to_batch_stride_zero"], 1)
        self.assertEqual(
            counters["inductor"]["cat_expand_outer_to_batch_stride_zero"], 1
        )
        self.assertEqual(len(lhs_metadata), 1)
        self.assertEqual(lhs_metadata[0][0][1:], (150, 1890))
        self.assertEqual(lhs_metadata[0][1], (0, 1890, 1))
        self._assert_rejection_counters(None)
        FileCheck().check_regex(r"reinterpret_tensor\([^\n]*\(0, 1890, 1\), 0\)").run(
            code
        )

    def test_production_topology_rank2_base(self):
        def fn(weights, rhs):
            lhs = torch.cat(
                [weight.expand(rhs.shape[0], -1, -1) for weight in weights], dim=1
            )
            lhs = lhs.expand(rhs.shape[0], lhs.shape[1], lhs.shape[2])
            rhs = rhs.expand(rhs.shape[0], rhs.shape[1], rhs.shape[2])
            return torch.bmm(lhs, rhs)

        weights = [
            torch.randn(10, 1890, device="cpu", dtype=torch.float16) for _ in range(15)
        ]
        rhs = torch.randn(2, 1890, 256, device="cpu", dtype=torch.float16)
        graph_metadata = []
        rewritten_lhs_metadata = []

        def record_graph_metadata(graph):
            bmm = next(
                iter(graph.find_nodes(op="call_function", target=aten.bmm.default))
            )
            outer_expand = bmm.args[0]
            cat = outer_expand.args[0]
            expands = cat.args[0]
            bases = [expand.args[0] for expand in expands]
            graph_metadata.append(
                {
                    "bases": [
                        (tuple(base.meta["val"].shape), base.meta["val"].stride())
                        for base in bases
                    ],
                    "expands": [
                        (
                            tuple(expand.meta["val"].shape[1:]),
                            expand.meta["val"].stride(),
                        )
                        for expand in expands
                    ],
                    "cat": (
                        tuple(cat.meta["val"].shape[1:]),
                        cat.meta["val"].stride(),
                    ),
                    "outer_expand": (
                        tuple(outer_expand.meta["val"].shape[1:]),
                        outer_expand.meta["val"].stride(),
                    ),
                    "rhs": (
                        bmm.args[1].target,
                        tuple(bmm.args[1].meta["val"].shape[1:]),
                        bmm.args[1].meta["val"].stride(),
                    ),
                    "bmm": (
                        tuple(bmm.meta["val"].shape[1:]),
                        bmm.meta["val"].stride(),
                    ),
                }
            )

        def record_rewritten_lhs_metadata(graph):
            bmm = next(
                iter(graph.find_nodes(op="call_function", target=aten.bmm.default))
            )
            lhs = bmm.args[0].meta["val"]
            rewritten_lhs_metadata.append((tuple(lhs.shape[1:]), lhs.stride()))

        torch._dynamo.reset()
        counters.clear()
        expected = fn(weights, rhs)
        torch._dynamo.mark_dynamic(rhs, 0)
        with inductor_config.patch(
            cat_expand_bmm_rewrite=True,
            max_autotune_gemm_backends="ATEN",
            post_grad_custom_pre_pass=record_graph_metadata,
            post_grad_custom_post_pass=record_rewritten_lhs_metadata,
        ):
            actual = torch.compile(fn, fullgraph=True)(weights, rhs)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["cat_expand_to_batch_stride_zero"], 1)
        self.assertEqual(
            counters["inductor"]["cat_expand_outer_to_batch_stride_zero"], 1
        )
        self._assert_rejection_counters(None)
        self.assertEqual(len(graph_metadata), 1)
        metadata = graph_metadata[0]
        self.assertEqual(
            metadata["bases"], [((10, 1890), (1890, 1)) for _ in range(15)]
        )
        self.assertEqual(
            metadata["expands"],
            [((10, 1890), (0, 1890, 1)) for _ in range(15)],
        )
        self.assertEqual(metadata["cat"], ((150, 1890), (283500, 1890, 1)))
        self.assertEqual(metadata["outer_expand"], ((150, 1890), (283500, 1890, 1)))
        self.assertEqual(
            metadata["rhs"],
            (aten.expand.default, (1890, 256), (483840, 256, 1)),
        )
        self.assertEqual(metadata["bmm"], ((150, 256), (38400, 256, 1)))
        self.assertEqual(rewritten_lhs_metadata, [((150, 1890), (0, 1890, 1))])

    def test_rewrite_disabled(self):
        def fn(weights, rhs):
            lhs = torch.cat(
                [
                    weight.unsqueeze(0).expand(rhs.shape[0], -1, -1)
                    for weight in weights
                ],
                dim=1,
            )
            return torch.bmm(lhs, rhs)

        weights = [
            torch.randn(rows, 8, device="cpu", dtype=torch.float16) for rows in (2, 3)
        ]
        rhs = torch.randn(4, 8, 6, device="cpu", dtype=torch.float16)
        lhs_strides = []

        def record_bmm_lhs_stride(graph):
            bmm = next(
                iter(graph.find_nodes(op="call_function", target=aten.bmm.default))
            )
            lhs_strides.append(bmm.args[0].meta["val"].stride())

        torch._dynamo.reset()
        counters.clear()
        expected = fn(weights, rhs)
        with inductor_config.patch(
            cat_expand_bmm_rewrite=False,
            max_autotune_gemm_backends="ATEN",
            post_grad_custom_post_pass=record_bmm_lhs_stride,
        ):
            actual = torch.compile(fn, fullgraph=True)(weights, rhs)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["cat_expand_to_batch_stride_zero"], 0)
        self.assertEqual(lhs_strides, [(40, 8, 1)])
        self._assert_rejection_counters(None)

    def test_rewrite_rejected_ck_backend(self):
        actual, expected, lhs_strides = self._run_admission_case(
            torch.float16, "ATEN,CK"
        )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["cat_expand_to_batch_stride_zero"], 0)
        self.assertNotEqual(lhs_strides[0][0], 0)
        self._assert_rejection_counters("backend")

    @parametrize("backend", ["CKTILE", "NVGEMM", "CUTEDSL", "FUTURE_BACKEND"])
    def test_rewrite_rejected_unsupported_backend(self, backend):
        actual, expected, lhs_strides = self._run_admission_case(
            torch.float16, f"ATEN,{backend}"
        )

        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["cat_expand_to_batch_stride_zero"], 0)
        self.assertNotEqual(lhs_strides[0][0], 0)
        self._assert_rejection_counters("backend")

    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_rewrite_rejected_unsupported_dtype(self, dtype):
        actual, expected, lhs_strides = self._run_admission_case(
            dtype, "ATEN,TRITON,CPP,CUTLASS"
        )

        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["cat_expand_to_batch_stride_zero"], 0)
        self.assertNotEqual(lhs_strides[0][0], 0)
        self._assert_rejection_counters("unsupported_dtype")

    def test_rewrite_allowed_backend_set(self):
        if not HAS_GPU:
            self.skipTest("CUDA backend admission requires a GPU")

        actual, expected, lhs_strides = self._run_admission_case(
            torch.float16,
            "ATEN,TRITON,CPP,CUTLASS",
            device=GPU_TYPE,
        )

        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["cat_expand_to_batch_stride_zero"], 1)
        self.assertEqual(lhs_strides[0][0], 0)
        self._assert_rejection_counters(None)

    def test_rewrite_rejected_cpp_backend_on_cpu(self):
        actual, expected, lhs_strides = self._run_admission_case(
            torch.float16,
            "CPP",
            rows=96,
            reduction=196,
            columns=84,
            batch=2,
            max_autotune=True,
        )

        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        self.assertEqual(counters["inductor"]["cat_expand_to_batch_stride_zero"], 0)
        self.assertNotEqual(lhs_strides[0][0], 0)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)
        self._assert_rejection_counters("backend")

    def _run_safe_gpu_backend(self, backend):
        """Compile and verify the rewrite for one GPU BMM backend."""

        if not HAS_GPU:
            self.skipTest("GPU is unavailable")
        if backend != "ATEN" and not is_big_gpu():
            self.skipTest(f"{backend} templates require a big GPU")

        def fn(weights, rhs):
            lhs = torch.cat(
                [weight.expand(rhs.shape[0], -1, -1) for weight in weights], dim=1
            )
            return torch.bmm(lhs, rhs)

        weights = [
            torch.randn(1, 32, 64, device=GPU_TYPE, dtype=torch.float16)
            for _ in range(2)
        ]
        rhs = torch.randn(2, 64, 64, device=GPU_TYPE, dtype=torch.float16)
        lhs_strides = []

        def record_bmm_lhs_stride(graph):
            bmm = next(
                iter(graph.find_nodes(op="call_function", target=aten.bmm.default))
            )
            lhs_strides.append(bmm.args[0].meta["val"].stride())

        torch._dynamo.reset()
        counters.clear()
        expected = fn(weights, rhs)
        with inductor_config.patch(
            cat_expand_bmm_rewrite=True,
            max_autotune=True,
            max_autotune_gemm_backends=backend,
            post_grad_custom_post_pass=record_bmm_lhs_stride,
        ):
            actual, (code,) = run_and_get_code(
                torch.compile(fn, fullgraph=True), weights, rhs
            )

        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        self.assertEqual(counters["inductor"]["cat_expand_to_batch_stride_zero"], 1)
        self.assertEqual(lhs_strides[0][0], 0)
        expected_code = {
            "ATEN": "extern_kernels.bmm",
            "TRITON": "triton_tem_fused_bmm",
            "CUTLASS": "cutlass_",
        }[backend]
        FileCheck().check(expected_code).run(code)

    @parametrize("backend", ["ATEN", "TRITON"])
    def test_rewrite_safe_common_gpu_backend(self, backend):
        self._run_safe_gpu_backend(backend)

    def test_rewrite_safe_cuda_cutlass_backend(self):
        if GPU_TYPE != "cuda" or torch.version.hip:
            self.skipTest("CUTLASS is unavailable on this GPU backend")
        from torch._inductor.codegen.cutlass.utils import try_import_cutlass

        if not try_import_cutlass():
            self.skipTest("CUTLASS is unavailable in this test environment")
        self._run_safe_gpu_backend("CUTLASS")

    @parametrize("device", [GPU_TYPE, "cpu"])
    @parametrize("direct_rank3", [False, True])
    def test_dynamic_batch(self, device, direct_rank3):
        if device != "cpu" and not HAS_GPU:
            self.skipTest("GPU is unavailable")

        def fn(weights, rhs):
            lhs = torch.cat(
                [
                    (weight if direct_rank3 else weight.unsqueeze(0)).expand(
                        rhs.shape[0], -1, -1
                    )
                    for weight in weights
                ],
                dim=1,
            )
            return torch.bmm(lhs, rhs)

        weight_shapes = [(rows, 8) for rows in (2, 3, 4)]
        if direct_rank3:
            weight_shapes = [(1, *shape) for shape in weight_shapes]
        weights = [
            torch.randn(shape, device=device, dtype=torch.float16)
            for shape in weight_shapes
        ]
        rhs = torch.randn(3, 8, 6, device=device, dtype=torch.float16)
        lhs_strides = []

        def record_bmm_lhs_stride(graph):
            for node in graph.find_nodes(op="call_function", target=aten.bmm.default):
                lhs_strides.append(node.args[0].meta["val"].stride())

        torch._dynamo.reset()
        counters.clear()
        torch._dynamo.mark_dynamic(rhs, 0)
        compiled = torch.compile(fn, fullgraph=True)
        with inductor_config.patch(
            cat_expand_bmm_rewrite=True,
            max_autotune_gemm_backends="ATEN",
            post_grad_custom_post_pass=record_bmm_lhs_stride,
        ):
            actual, (code,) = run_and_get_code(compiled, weights, rhs)
            second_rhs = torch.randn(5, 8, 6, device=device, dtype=torch.float16)
            second_actual = compiled(weights, second_rhs)

        torch.testing.assert_close(actual, fn(weights, rhs))
        torch.testing.assert_close(second_actual, fn(weights, second_rhs))
        self.assertEqual(counters["inductor"]["cat_expand_to_batch_stride_zero"], 1)
        self.assertEqual(len(lhs_strides), 1)
        self.assertEqual(lhs_strides[0][0], 0)
        FileCheck().check_regex(r"reinterpret_tensor\([^\n]*\(0, 8, 1\), 0\)").run(code)

    @parametrize("outer_expand", [False, True])
    def test_rewrite_rejected_rhs_alias(self, outer_expand):
        def fn(weights):
            lhs = torch.cat(
                [weight.unsqueeze(0).expand(3, -1, -1) for weight in weights],
                dim=1,
            )
            if outer_expand:
                lhs = lhs.expand(3, lhs.shape[1], lhs.shape[2])
            return torch.bmm(lhs, lhs)

        weights = [
            torch.randn(2, 4, device="cpu", dtype=torch.float16) for _ in range(2)
        ]

        torch._dynamo.reset()
        counters.clear()
        expected = fn(weights)
        with inductor_config.patch(
            cat_expand_bmm_rewrite=True,
            max_autotune_gemm_backends="ATEN",
        ):
            actual = torch.compile(fn, fullgraph=True)(weights)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["cat_expand_to_batch_stride_zero"], 0)
        self.assertEqual(
            counters["inductor"]["cat_expand_outer_to_batch_stride_zero"], 0
        )
        self._assert_rejection_counters("users")

    def test_rewrite_rejected_duplicate_expand(self):
        def fn(weight, rhs):
            expands = [
                weight.unsqueeze(0).expand(rhs.shape[0], -1, -1) for _ in range(2)
            ]
            lhs = torch.cat(expands, dim=1)
            return torch.bmm(lhs, rhs)

        weight = torch.randn(2, 4, device="cpu", dtype=torch.float16)
        rhs = torch.randn(3, 4, 5, device="cpu", dtype=torch.float16)
        duplicate_inputs = []

        def force_duplicate_expand(graph):
            bmm = next(
                iter(graph.find_nodes(op="call_function", target=aten.bmm.default))
            )
            cat = bmm.args[0]
            expands = cat.args[0]
            cat.update_arg(0, [expands[0], expands[0]])
            duplicate_inputs.append(cat.args[0][0] is cat.args[0][1])

        torch._dynamo.reset()
        counters.clear()
        expected = fn(weight, rhs)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            with inductor_config.patch(
                cat_expand_bmm_rewrite=True,
                max_autotune_gemm_backends="ATEN",
                post_grad_custom_pre_pass=force_duplicate_expand,
            ):
                actual = torch.compile(fn, fullgraph=True)(weight, rhs)

        erase_warnings = [
            str(warning.message)
            for warning in caught_warnings
            if "on an already erased node" in str(warning.message)
        ]
        self.assertEqual(duplicate_inputs, [True])
        torch.testing.assert_close(actual, expected)
        self.assertEqual(erase_warnings, [])
        self.assertEqual(counters["inductor"]["cat_expand_to_batch_stride_zero"], 0)
        self._assert_rejection_counters("users")

    @parametrize(
        "case",
        [
            "batch_cat",
            "non_batch_expand",
            "non_batch_unsqueeze",
            "multiple_consumers",
            "expand_multiple_consumers",
            "outer_expand_broadcast",
            "outer_expand_multiple_consumers",
            "non_bmm_consumer",
        ],
    )
    def test_rewrite_rejected(self, case):
        expected_rejection = {
            "batch_cat": None,
            "non_batch_expand": "shape_stride_identity",
            "non_batch_unsqueeze": "batch_one_base",
            "multiple_consumers": None,
            "outer_expand_broadcast": "shape_stride_identity",
            "outer_expand_multiple_consumers": None,
            "expand_multiple_consumers": None,
            "non_bmm_consumer": None,
        }[case]
        if case == "batch_cat":
            weights = [
                torch.randn(2, 5, device="cpu", dtype=torch.float16) for _ in range(2)
            ]
            rhs = torch.randn(4, 5, 3, device="cpu", dtype=torch.float16)

            def fn(weights, rhs):
                batch = rhs.shape[0] // len(weights)
                lhs = torch.cat(
                    [weight.unsqueeze(0).expand(batch, -1, -1) for weight in weights],
                    dim=0,
                )
                return torch.bmm(lhs, rhs)

        elif case == "non_batch_expand":
            weights = [
                torch.randn(1, 5, device="cpu", dtype=torch.float16) for _ in range(2)
            ]
            rhs = torch.randn(3, 5, 4, device="cpu", dtype=torch.float16)

            def fn(weights, rhs):
                lhs = torch.cat(
                    [
                        weight.unsqueeze(0).expand(rhs.shape[0], 2, -1)
                        for weight in weights
                    ],
                    dim=1,
                )
                return torch.bmm(lhs, rhs)

        elif case == "non_batch_unsqueeze":
            weights = [
                torch.randn(3, 5, device="cpu", dtype=torch.float16) for _ in range(2)
            ]
            rhs = torch.randn(3, 5, 4, device="cpu", dtype=torch.float16)

            def fn(weights, rhs):
                lhs = torch.cat(
                    [weight.unsqueeze(1).expand(-1, 2, -1) for weight in weights],
                    dim=1,
                )
                return torch.bmm(lhs, rhs)

        elif case == "outer_expand_broadcast":
            weights = [
                torch.randn(2, 5, device="cpu", dtype=torch.float16) for _ in range(2)
            ]
            rhs = torch.randn(3, 5, 4, device="cpu", dtype=torch.float16)

            def fn(weights, rhs):
                lhs = torch.cat(
                    [weight.unsqueeze(0).expand(1, -1, -1) for weight in weights],
                    dim=1,
                )
                lhs = lhs.expand(rhs.shape[0], lhs.shape[1], lhs.shape[2])
                return torch.bmm(lhs, rhs)

        else:
            weights = [
                torch.randn(2, 5, device="cpu", dtype=torch.float16) for _ in range(2)
            ]
            rhs = torch.randn(3, 5, 4, device="cpu", dtype=torch.float16)

            def fn(weights, rhs):
                expands = [
                    weight.unsqueeze(0).expand(rhs.shape[0], -1, -1)
                    for weight in weights
                ]
                lhs = torch.cat(expands, dim=1)
                if case == "multiple_consumers":
                    return torch.bmm(lhs, rhs), lhs
                if case == "expand_multiple_consumers":
                    return torch.bmm(lhs, rhs), expands[0]
                if case == "outer_expand_multiple_consumers":
                    lhs = lhs.expand(rhs.shape[0], lhs.shape[1], lhs.shape[2])
                    return torch.bmm(lhs, rhs), lhs
                return lhs + 1

        torch._dynamo.reset()
        counters.clear()
        expected = fn(weights, rhs)
        with inductor_config.patch(
            cat_expand_bmm_rewrite=True,
            max_autotune_gemm_backends="ATEN",
        ):
            actual = torch.compile(fn, fullgraph=True)(weights, rhs)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["cat_expand_to_batch_stride_zero"], 0)
        self._assert_rejection_counters(expected_rejection)

    @parametrize(
        "metadata_case",
        [
            "missing",
            "rank",
            "rank_expand",
            "rank_base",
            "dtype",
            "layout",
            "unsupported_device",
            "cat_dimension",
            "bmm_compatibility",
        ],
    )
    def test_rewrite_rejected_metadata(self, metadata_case):
        def fn(weights, rhs):
            lhs = torch.cat(
                [weight.expand(rhs.shape[0], -1, -1) for weight in weights], dim=1
            )
            return torch.bmm(lhs, rhs)

        weights = [
            torch.randn(1, 2, 5, device="cpu", dtype=torch.float16) for _ in range(2)
        ]
        rhs = torch.randn(3, 5, 4, device="cpu", dtype=torch.float16)
        saved_args = {}
        saved_values = {}

        def mutate_metadata(graph):
            bmm = next(
                iter(graph.find_nodes(op="call_function", target=aten.bmm.default))
            )
            if metadata_case == "unsupported_device":
                cat = bmm.args[0]
                expands = cat.args[0]
                matched_nodes = [
                    cat,
                    bmm.args[1],
                    *expands,
                    *(expand.args[0] for expand in expands),
                ]
                with unset_fake_temporarily():
                    for matched_node in matched_nodes:
                        value = matched_node.meta["val"]
                        saved_values[matched_node] = value
                        matched_node.meta["val"] = torch.empty_strided(
                            tuple(value.shape),
                            tuple(value.stride()),
                            dtype=value.dtype,
                            device="meta",
                        )
                return
            node = bmm.args[1] if metadata_case == "bmm_compatibility" else bmm.args[0]
            if metadata_case in ("rank_expand", "rank_base"):
                node = node.args[0][0]
            if metadata_case == "rank_base":
                node = node.args[0]
            if metadata_case == "cat_dimension":
                saved_args[node] = node.args
                node.update_arg(1, 0)
                return
            saved_values[node] = node.meta["val"]
            value = saved_values[node]
            if metadata_case == "missing":
                del node.meta["val"]
            elif metadata_case in ("rank", "rank_expand"):
                node.meta["val"] = value[0]
            elif metadata_case == "rank_base":
                node.meta["val"] = value[0, 0]
            elif metadata_case == "dtype":
                node.meta["val"] = value.to(torch.float64)
            elif metadata_case == "layout":
                with unset_fake_temporarily():
                    node.meta["val"] = torch.empty(
                        tuple(value.shape),
                        dtype=value.dtype,
                        device=value.device,
                        layout=torch.sparse_coo,
                    )
            else:
                node.meta["val"] = value.new_empty(
                    (value.shape[0], value.shape[1] + 1, value.shape[2])
                )

        def restore_metadata(_graph):
            for node, args in saved_args.items():
                node.args = args
            for node, value in saved_values.items():
                node.meta["val"] = value

        expected_rejection = {
            "missing": "topology",
            "rank": "rank",
            "rank_expand": "rank",
            "rank_base": "rank",
            "dtype": "dtype_device_layout",
            "layout": "dtype_device_layout",
            "unsupported_device": "unsupported_device",
            "cat_dimension": "cat_dimension",
            "bmm_compatibility": "bmm_compatibility",
        }[metadata_case]
        expected_rank_detail = {
            "rank": "cat_rhs",
            "rank_expand": "expand",
            "rank_base": "base",
        }.get(metadata_case)
        torch._dynamo.reset()
        counters.clear()
        expected = fn(weights, rhs)
        with inductor_config.patch(
            cat_expand_bmm_rewrite=True,
            max_autotune_gemm_backends="ATEN",
            post_grad_custom_pre_pass=mutate_metadata,
            post_grad_custom_post_pass=restore_metadata,
        ):
            actual = torch.compile(fn, fullgraph=True)(weights, rhs)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["cat_expand_to_batch_stride_zero"], 0)
        self._assert_rejection_counters(expected_rejection, expected_rank_detail)


if __name__ == "__main__":
    run_tests()
