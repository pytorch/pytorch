# Owner(s): ["module: dynamo"]

import os
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.config
from torch._dynamo import debug_utils
from torch._dynamo.debug_utils import aot_graph_input_parser, generate_env_vars_string
from torch._dynamo.test_case import TestCase
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_device_type import instantiate_device_type_tests


f32 = torch.float32
i64 = torch.int64
i32 = torch.int32


class TestDebugUtils(TestCase):
    def test_cast_model_to_fp64_dtype_args(self):
        # Test that dtype arguments are converted to fp64

        def fn(x):
            return (
                torch.ops.prims.convert_element_type(x, torch.float16),
                x.to(torch.float16),
                torch.full(x.shape, 2, dtype=torch.float32, device=x.device),
                x.new_empty(x.shape),
            )

        x = torch.randn(32, device="cpu")
        decomps = torch._decomp.core_aten_decompositions()
        fx = make_fx(fn, decomposition_table=decomps)(x)

        self.assertExpectedInline(
            fx.code.lstrip(),
            """\
def forward(self, x_1):
    convert_element_type = torch.ops.prims.convert_element_type.default(x_1, torch.float16)
    _to_copy = torch.ops.aten._to_copy.default(x_1, dtype = torch.float16);  x_1 = None
    full = torch.ops.aten.full.default([32], 2, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    empty = torch.ops.aten.empty.memory_format([32], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    return (convert_element_type, _to_copy, full, empty)
    """,  # NOQA: B950
        )

        _, fp64_examples = debug_utils.cast_to_fp64(fx, (x,))
        self.assertEqual(fp64_examples, (x.to(torch.float64),))

        self.assertExpectedInline(
            fx.code.lstrip(),
            """\
def forward(self, x_1):
    convert_element_type = torch.ops.prims.convert_element_type.default(x_1, torch.float64)
    _to_copy = torch.ops.aten._to_copy.default(x_1, dtype = torch.float64);  x_1 = None
    full = torch.ops.aten.full.default([32], 2, dtype = torch.float64, device = device(type='cpu'), pin_memory = False)
    empty = torch.ops.aten.empty.memory_format([32], dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    return (convert_element_type, _to_copy, full, empty)
    """,  # NOQA: B950
        )

    @patch.dict(os.environ, {"TORCHINDUCTOR_MAX_AUTOTUNE": "1", "TEST_ENV": "1"})
    def test_generate_env_vars_string(self):
        env_strings = generate_env_vars_string()
        self.assertIn(
            """os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '1'
""",
            env_strings,
        )
        self.assertIn(
            """import os
""",
            env_strings,
        )
        self.assertNotIn(
            """TEST_ENV
""",
            env_strings,
        )


class TestDebugUtilsDevice(TestCase):
    def test_aot_graph_parser(self, device):
        def forward(
            self,
            primals_1: "f32[1001, 6]",
            primals_2: "f32[1001]",
            primals_3: "f32[1001, 64]",
            primals_4: "f32[4190]",
            primals_5: "f32[4190]",
            primals_6: "f32[1739, 4190]",
            primals_48: "f32[6144, 4191]",
        ):
            _tensor_constant0: "i64[4190]" = self._tensor_constant0
            lift_fresh_copy: "i64[4190]" = torch.ops.aten.lift_fresh_copy.default(
                _tensor_constant0
            )
            _tensor_constant0 = None
            index: "f32[6144, 4190]" = torch.ops.aten.index.Tensor(  # noqa: F841
                primals_48, [None, lift_fresh_copy]
            )
            lift_fresh_copy = None

            _tensor_constant1: "i64[6]" = self._tensor_constant1
            lift_fresh_copy_1: "i64[6]" = torch.ops.aten.lift_fresh_copy.default(
                _tensor_constant1
            )
            _tensor_constant1 = None
            index_1: "f32[6144, 6]" = torch.ops.aten.index.Tensor(
                primals_48, [None, lift_fresh_copy_1]
            )
            primals_48 = lift_fresh_copy_1 = None
            permute: "f32[6, 1001]" = torch.ops.aten.permute.default(primals_1, [1, 0])
            primals_1 = None
            addmm: "f32[6144, 1001]" = torch.ops.aten.addmm.default(
                primals_2, index_1, permute
            )
            primals_2 = permute = None
            amax: "f32[6144, 1]" = torch.ops.aten.amax.default(addmm, [-1], True)
            sub: "f32[6144, 1001]" = torch.ops.aten.sub.Tensor(addmm, amax)
            exp: "f32[6144, 1001]" = torch.ops.aten.exp.default(sub)
            sub = None
            sum_1: "f32[6144, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
            div: "f32[6144, 1001]" = torch.ops.aten.div.Tensor(exp, sum_1)
            exp = None

            full_default: "i32[6144, 1001]" = torch.ops.aten.full.default(
                [6144, 1001],
                1,
                dtype=torch.int32,
                layout=torch.strided,
                device=device,
                pin_memory=False,
            )

            iota: "i32[1001]" = torch.ops.prims.iota.default(
                1001,
                start=0,
                step=1,
                dtype=torch.int32,
                device=device,
                requires_grad=False,
            )

            mul: "i32[6144, 1001]" = torch.ops.aten.mul.Tensor(full_default, iota)
            full_default = iota = None

            iota_1: "i32[6144]" = torch.ops.prims.iota.default(
                6144,
                start=0,
                step=1001,
                dtype=torch.int32,
                device=device,
                requires_grad=False,
            )
            view: "i32[6150144]" = torch.ops.aten.reshape.default(mul, [-1])
            mul = None
            view_1: "f32[6150144]" = torch.ops.aten.reshape.default(div, [-1])
            div = None
            _embedding_bag = torch.ops.aten._embedding_bag.default(
                primals_3, view, iota_1, False, 0, False, view_1
            )

            return _embedding_bag

        kwargs = aot_graph_input_parser(forward, device=device)
        # runs successfully
        forward(**kwargs)

    def test_sym_aot_graph_parser(self, device):
        def forward(
            self,
            primals_1: "f32[1001, 6]",  # noqa: F821
            primals_2: "f32[s0]",  # noqa: F821
            primals_3: "Sym(s0)",  # noqa: F821,
            primals_4: "f32[s1]",  # noqa: F821,
            primals_5: "Sym(s1)",  # noqa: F821,
        ):
            _tensor_constant0: "i64[4190]" = self._tensor_constant0

        kwargs = aot_graph_input_parser(
            forward, device=device, sym_shapes={"s0": 10}, default_sym_shape=5
        )

        self.assertEqual(list(kwargs["primals_2"].shape), [10])
        self.assertEqual(kwargs["primals_3"], 10)

        self.assertEqual(list(kwargs["primals_4"].shape), [5])
        self.assertEqual(kwargs["primals_5"], 5)


instantiate_device_type_tests(TestDebugUtils, globals())

devices = ["cuda", "hpu"]
instantiate_device_type_tests(TestDebugUtilsDevice, globals(), only_for=devices)


class TestBackendOverrideIntegration(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self._backends_called = []

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    def _fn_with_4_graphs(self, x):
        x = x + 1
        torch._dynamo.graph_break()
        x = x * 2
        torch._dynamo.graph_break()
        x = x - 1
        torch._dynamo.graph_break()
        x = x / 2
        return x

    def _run_with_override(self, device, override_config, default_backend="eager"):
        from torch._dynamo.graph_id_filter import lookup_backend_with_mode

        torch._dynamo.reset()
        self._backends_called.clear()
        original_lookup = lookup_backend_with_mode

        def tracking_lookup(backend_str):
            self._backends_called.append(backend_str)
            return original_lookup(backend_str)

        with (
            patch.object(
                torch._dynamo.config, "debug_backend_override", override_config
            ),
            patch(
                "torch._dynamo.graph_id_filter.lookup_backend_with_mode",
                tracking_lookup,
            ),
        ):
            compiled_fn = torch.compile(self._fn_with_4_graphs, backend=default_backend)
            compiled_fn(torch.randn(10, device=device))

        return self._backends_called.copy()

    def test_no_override(self, device):
        result = self._run_with_override(device, "")
        self.assertEqual(result, [])

    def test_override_all_graphs(self, device):
        result = self._run_with_override(device, ">=0:aot_eager")
        self.assertEqual(result, ["aot_eager", "aot_eager", "aot_eager", "aot_eager"])

    def test_override_greater_than(self, device):
        result = self._run_with_override(device, ">0:eager")
        self.assertEqual(result, ["eager", "eager", "eager"])

    def test_override_less_than(self, device):
        result = self._run_with_override(device, "<2:aot_eager")
        self.assertEqual(result, ["aot_eager", "aot_eager"])

    def test_override_less_or_equal(self, device):
        result = self._run_with_override(device, "<=1:aot_eager")
        self.assertEqual(result, ["aot_eager", "aot_eager"])

    def test_override_single_id(self, device):
        result = self._run_with_override(device, "1:aot_eager")
        self.assertEqual(result, ["aot_eager"])

    def test_override_multiple_ids(self, device):
        result = self._run_with_override(device, "0,2:aot_eager")
        self.assertEqual(result, ["aot_eager", "aot_eager"])

    def test_override_range(self, device):
        result = self._run_with_override(device, "1-2:eager")
        self.assertEqual(result, ["eager", "eager"])

    def test_multiple_rules(self, device):
        result = self._run_with_override(device, "0:aot_eager;1:inductor;3:eager")
        self.assertEqual(result, ["aot_eager", "inductor", "eager"])

    def test_first_rule_wins(self, device):
        result = self._run_with_override(device, ">=0:aot_eager;>=1:inductor")
        self.assertEqual(result, ["aot_eager", "aot_eager", "aot_eager", "aot_eager"])

    def test_complex_config(self, device):
        result = self._run_with_override(device, "0:aot_eager;>=2:inductor")
        self.assertEqual(result, ["aot_eager", "inductor", "inductor"])

    def test_inductor_with_mode(self, device):
        result = self._run_with_override(device, ">=0:inductor:reduce-overhead")
        self.assertEqual(
            result,
            [
                "inductor:reduce-overhead",
                "inductor:reduce-overhead",
                "inductor:reduce-overhead",
                "inductor:reduce-overhead",
            ],
        )


instantiate_device_type_tests(
    TestBackendOverrideIntegration, globals(), only_for=["cpu", "cuda"]
)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
