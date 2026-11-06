# Owner(s): ["module: dynamo"]

import ast
import itertools
import json
import math
import os
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.config
from torch._dynamo import debug_utils
from torch._dynamo.debug_utils import (
    _serialize_sym_expr,
    aot_graph_input_parser,
    generate_env_vars_string,
    InputReader,
    InputWriter,
    NNModuleToString,
)
from torch._dynamo.test_case import TestCase
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    TemporaryDirectoryName,
)
from torch.utils._content_store import (
    ContentStoreReader,
    ContentStoreWriter,
    zstd_available,
)


f32 = torch.float32
i64 = torch.int64
i32 = torch.int32


class TestDebugUtils(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_serialize_symbolic_storage_nbytes(self):
        from sympy import floor

        from torch._dynamo.source import ConstantSource
        from torch.fx.experimental.sym_node import SymNode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from torch.utils._sympy.functions import Max

        shape_env = ShapeEnv()
        symbol = shape_env.create_symbol(4, ConstantSource("storage_size"))
        expr = 4 * symbol + 18428 * Max(1, symbol)
        nbytes = torch.SymInt(SymNode(expr, shape_env, int, hint=73728))
        source = _serialize_sym_expr(nbytes)

        self.assertNotIn("Max", source)
        for value in (0, 1, 4):
            self.assertEqual(
                eval(source, {"max": max}, {str(symbol): value}),
                int(expr.subs(symbol, value)),
            )

        floor_expr = floor(symbol / 2)
        floor_nbytes = torch.SymInt(SymNode(floor_expr, shape_env, int, hint=2))
        floor_source = _serialize_sym_expr(floor_nbytes)
        self.assertIn("math.floor", floor_source)
        self.assertEqual(
            eval(floor_source, {"math": math}, {str(symbol): 4}),
            2,
        )

    def test_input_writer_symbolic_tensor_metadata_is_python(self):
        from torch._dynamo.debug_utils import InputWriter
        from torch._dynamo.source import ConstantSource
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.sym_node import SymNode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from torch.utils._sympy.functions import CeilToInt, IntTrueDiv

        shape_env = ShapeEnv()
        symbol = shape_env.create_symbol(64, ConstantSource("s33"))
        # 32*ceil(s33/32) reprs as 32*CeilToInt(IntTrueDiv(s33, 32)) under sympy
        expr = 32 * CeilToInt(IntTrueDiv(symbol, 32))
        dim = torch.SymInt(SymNode(expr, shape_env, int, hint=64))

        writer = InputWriter(None)
        with FakeTensorMode(shape_env=shape_env):
            base = torch.empty(4, dim, device="cpu")
            writer.tensor("shape_arg", base)
            writer.tensor("stride_arg", base.transpose(0, 1))
            writer.tensor("offset_arg", base[1])

        source = "\n".join(writer.lines())
        self.assertIn("math.ceil", source)
        self.assertNotIn("CeilToInt", source)
        self.assertNotIn("IntTrueDiv", source)

        class Reader:
            def __init__(self):
                self.tensors = []

            def storage(self, storage_hash, nbytes, **kwargs):
                return nbytes

            def tensor(self, buf, shape, stride=None, *, storage_offset=0, **kwargs):
                self.tensors.append((shape, stride, storage_offset))

        # The repro preamble imports math and torch (sym_max shows up in
        # contiguous strides); the symbol is bound by the caller.
        namespace = {"math": math, "torch": torch, str(symbol): 64}
        exec(source, namespace)
        reader = Reader()
        namespace["load_args"](reader)
        self.assertEqual(
            reader.tensors,
            [((4, 64), None, 0), ((64, 4), (1, 64), 0), ((64,), None, 64)],
        )

    def test_repro_templates_import_symexpr_dependencies(self):
        from torch._dynamo.repro.after_aot import generate_compiler_repro_string
        from torch._dynamo.repro.after_dynamo import generate_dynamo_fx_repro_string

        gm = torch.fx.symbolic_trace(lambda x: x + 1)
        args = [torch.ones(1)]

        repros = {
            "after_aot": generate_compiler_repro_string(gm, args),
            "after_dynamo": generate_dynamo_fx_repro_string(
                gm, args, compiler_name="eager"
            ),
        }
        for name, source in repros.items():
            with self.subTest(name=name):
                tree = ast.parse(source)
                imports = ast.Module(
                    body=[
                        node
                        for node in tree.body
                        if isinstance(node, (ast.Import, ast.ImportFrom))
                    ],
                    type_ignores=[],
                )
                namespace = {}
                exec(compile(imports, f"<{name}>", "exec"), namespace)
                self.assertEqual(eval("math.floor(3 / 2)", namespace), 1)

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
    """,
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
    """,
        )

    def test_cast_model_to_fp64_disables_half_to_float(self):
        for op_name, overload, use_kwargs in itertools.product(
            ("_softmax", "_log_softmax"),
            ("default", "out"),
            (False, True),
        ):
            with self.subTest(
                op_name=op_name, overload=overload, use_kwargs=use_kwargs
            ):
                graph = torch.fx.Graph()
                x = graph.placeholder("x")
                op = getattr(getattr(torch.ops.aten, op_name), overload)
                kwargs = {}
                inputs = [torch.randn(2, 4, dtype=torch.float16)]
                if overload == "out":
                    kwargs["out"] = graph.placeholder("out")
                    inputs.append(torch.empty(2, 4, dtype=torch.float16))
                if use_kwargs:
                    kwargs["half_to_float"] = True
                    output = graph.call_function(op, args=(x, 1), kwargs=kwargs)
                else:
                    output = graph.call_function(op, args=(x, 1, True), kwargs=kwargs)
                graph.output(output)
                model = torch.fx.GraphModule({}, graph)

                fp64_model, fp64_inputs = debug_utils.cast_to_fp64(model, inputs)
                softmax = next(
                    node
                    for node in fp64_model.graph.nodes
                    if node.op == "call_function"
                )
                half_to_float = (
                    softmax.kwargs["half_to_float"] if use_kwargs else softmax.args[2]
                )

                self.assertFalse(half_to_float)
                self.assertEqual(fp64_model(*fp64_inputs).dtype, torch.float64)

    @patch.dict(
        os.environ,
        {
            "TORCHINDUCTOR_MAX_AUTOTUNE": "1",
            "TEST_ENV": "1",
            "TORCHINDUCTOR_ENV_SINGLE_QUOTES": "inductor_'env'",
            "TORCHINDUCTOR_ENV_DOUBLE_QUOTES": 'inductor_"env"',
            "TORCHDYNAMO_REPRO_AFTER": "aot",
            "TORCHDYNAMO_REPRO_LEVEL": "4",
        },
    )
    def test_generate_env_vars_string(self):
        env_strings = generate_env_vars_string()
        self.assertIn(
            """os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '1'
""",
            env_strings,
        )
        self.assertIn(
            """os.environ['TORCHINDUCTOR_ENV_SINGLE_QUOTES'] = 'inductor_"env"'
""",
            env_strings,
        )
        self.assertIn(
            """os.environ['TORCHINDUCTOR_ENV_DOUBLE_QUOTES'] = 'inductor_"env"'
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
        self.assertNotIn("os.environ['TORCHDYNAMO_REPRO_AFTER']", env_strings)
        self.assertNotIn("os.environ['TORCHDYNAMO_REPRO_LEVEL']", env_strings)
        self.assertIn("os.environ.pop('TORCHDYNAMO_REPRO_AFTER', None)", env_strings)
        self.assertIn("os.environ.pop('TORCHDYNAMO_REPRO_LEVEL', None)", env_strings)

    def test_cuda_system_info_comment_nvcc_os_errors(self):
        self.addCleanup(debug_utils._cuda_system_info_comment.cache_clear)
        errors = (
            PermissionError(13, "Permission denied", "nvcc"),
            OSError(5, "Input/output error", "nvcc"),
        )

        for error in errors:
            with self.subTest(error=type(error).__name__):
                debug_utils._cuda_system_info_comment.cache_clear()
                with (
                    patch.object(torch.cuda, "is_available", return_value=True),
                    patch.object(torch.version, "hip", None),
                    patch.object(
                        debug_utils.subprocess,
                        "check_output",
                        side_effect=error,
                    ),
                    patch.object(torch.cuda, "device_count", return_value=0),
                ):
                    result = debug_utils._cuda_system_info_comment()

                self.assertIn("# nvcc not found\n", result)
                self.assertIn("# GPU Hardware Info: \n", result)


class TestDebugUtilsDevice(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

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
            primals_1: "f32[1001, 6]",
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


class TestNNModuleToStringBufferDevice(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_nn_module_to_string_buffer_device(self, device):
        gm = torch.fx.symbolic_trace(torch.nn.Identity())
        gm.register_buffer("test_buf", torch.empty(5, device=device))

        result = NNModuleToString.convert(gm)

        if torch.device(device).type == "cpu":
            self.assertNotIn(".to(", result)
        else:
            expected_device = str(torch.empty(1, device=device).device)
            self.assertIn(f'.to("{expected_device}")', result)
            self.assertNotIn(f".{self.device_type}()", result)

    def test_nn_module_to_string_param_device(self, device):
        gm = torch.fx.symbolic_trace(torch.nn.Identity())
        gm.register_parameter(
            "test_param", torch.nn.Parameter(torch.empty(5, device=device))
        )

        result = NNModuleToString.convert(gm)

        if torch.device(device).type == "cpu":
            self.assertNotIn("device=", result)
        else:
            expected_device = str(torch.empty(1, device=device).device)
            self.assertIn(f'device="{expected_device}"', result)
            self.assertNotIn(f', device="{self.device_type}")', result)


instantiate_device_type_tests(
    TestNNModuleToStringBufferDevice, globals(), allow_xpu=True
)

instantiate_device_type_tests(
    TestDebugUtilsDevice, globals(), except_for="mps", allow_xpu=True
)


class TestBackendOverrideIntegration(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

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
        from torch._dynamo.graph_id_filter import (
            _create_backend_router,
            get_backend_override_for_compile_id,
        )

        torch._dynamo.reset()
        # Clear the router cache to ensure fresh routers for each test
        _create_backend_router.cache_clear()
        self._backends_called.clear()
        original_get_override = get_backend_override_for_compile_id

        # Pre-parse the config to build a mapping of graph_id -> backend_str
        # by using the same parsing logic but extracting the original strings
        backend_str_map: dict[int, str] = {}
        if override_config:
            for rule_str in override_config.split(";"):
                rule_str = rule_str.strip()
                if not rule_str or ":" not in rule_str:
                    continue
                colon_idx = rule_str.find(":")
                filter_str = rule_str[:colon_idx].strip()
                backend_str = rule_str[colon_idx + 1 :].strip()
                # Parse the filter to extract graph IDs
                from torch._dynamo.graph_id_filter import GraphIdFilter

                gf = GraphIdFilter(filter_str)
                # Store the backend_str for any graph that matches this filter
                for graph_id in range(100):  # Check first 100 graphs
                    if graph_id in gf and graph_id not in backend_str_map:
                        backend_str_map[graph_id] = backend_str

        def tracking_get_override(compile_id, config_str):
            result = original_get_override(compile_id, config_str)
            if result is not None:
                graph_id = compile_id.frame_id
                if graph_id in backend_str_map:
                    self._backends_called.append(backend_str_map[graph_id])
            return result

        with (
            patch.object(
                torch._dynamo.config, "debug_backend_override", override_config
            ),
            patch(
                "torch._dynamo.output_graph.get_backend_override_for_compile_id",
                tracking_get_override,
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

    def test_conflicting_rules_raise(self, device):
        with self.assertRaisesRegex(
            torch._dynamo.exc.InternalTorchDynamoError,
            "Conflicting backend override",
        ):
            self._run_with_override(device, ">=0:aot_eager;>=1:inductor")

    def test_complex_config(self, device):
        result = self._run_with_override(device, "0:aot_eager;>=2:inductor")
        self.assertEqual(result, ["aot_eager", "inductor", "inductor"])

    def test_override_with_backward(self, device):
        """Verify that backend override works when backward compilation occurs."""
        from torch._dynamo.graph_id_filter import (
            _create_backend_router,
            get_backend_override_for_compile_id,
        )

        torch._dynamo.reset()
        _create_backend_router.cache_clear()
        overrides_applied = []
        original_get_override = get_backend_override_for_compile_id

        def tracking_get_override(compile_id, config_str):
            result = original_get_override(compile_id, config_str)
            if result is not None:
                overrides_applied.append(compile_id.frame_id)
            return result

        def fn(x):
            return (x * 2 + 1).sum()

        with (
            patch.object(
                torch._dynamo.config, "debug_backend_override", ">=0:aot_eager"
            ),
            patch(
                "torch._dynamo.output_graph.get_backend_override_for_compile_id",
                tracking_get_override,
            ),
        ):
            compiled_fn = torch.compile(fn, backend="eager")
            x = torch.randn(10, device=device, requires_grad=True)
            result = compiled_fn(x)
            result.backward()

        self.assertEqual(overrides_applied, [0])
        self.assertIsNotNone(x.grad)


instantiate_device_type_tests(
    TestBackendOverrideIntegration, globals(), except_for="mps"
)


class TestInductorConfigOverrideIntegration(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    def test_config_router_single_graph(self, device):
        from torch._dynamo.graph_id_filter import GraphConfigRouter

        router = GraphConfigRouter("0:triton.cudagraph_skip_dynamic_graphs=False")
        self.assertEqual(
            router.get_value_for_graph(0),
            {"triton.cudagraph_skip_dynamic_graphs": False},
        )
        self.assertIsNone(router.get_value_for_graph(1))

    def test_config_router_multiple_options(self, device):
        from torch._dynamo.graph_id_filter import GraphConfigRouter

        router = GraphConfigRouter(
            "0:triton.cudagraphs=False,triton.cudagraph_trees=False"
        )
        self.assertEqual(
            router.get_value_for_graph(0),
            {"triton.cudagraphs": False, "triton.cudagraph_trees": False},
        )

    def test_config_router_comparison(self, device):
        from torch._dynamo.graph_id_filter import GraphConfigRouter

        router = GraphConfigRouter(">1:triton.cudagraphs=True")
        self.assertIsNone(router.get_value_for_graph(0))
        self.assertIsNone(router.get_value_for_graph(1))
        self.assertEqual(router.get_value_for_graph(2), {"triton.cudagraphs": True})

    def test_config_router_range(self, device):
        from torch._dynamo.graph_id_filter import GraphConfigRouter

        router = GraphConfigRouter("1-3:triton.cudagraphs=False")
        self.assertIsNone(router.get_value_for_graph(0))
        self.assertEqual(router.get_value_for_graph(1), {"triton.cudagraphs": False})
        self.assertEqual(router.get_value_for_graph(2), {"triton.cudagraphs": False})
        self.assertEqual(router.get_value_for_graph(3), {"triton.cudagraphs": False})
        self.assertIsNone(router.get_value_for_graph(4))

    def test_config_router_value_types(self, device):
        from torch._dynamo.graph_id_filter import GraphConfigRouter

        router = GraphConfigRouter(
            "0:bool_opt=True,int_opt=42,float_opt=3.14,str_opt=hello,none_opt=None"
        )
        config = router.get_value_for_graph(0)
        self.assertEqual(config["bool_opt"], True)
        self.assertEqual(config["int_opt"], 42)
        self.assertAlmostEqual(config["float_opt"], 3.14)
        self.assertEqual(config["str_opt"], "hello")
        self.assertIsNone(config["none_opt"])

    def test_config_router_aggregation(self, device):
        from torch._dynamo.graph_id_filter import GraphConfigRouter

        router = GraphConfigRouter("0:a=1;>=0:b=2")
        # Graph 0 matches both rules, configs are merged
        self.assertEqual(router.get_value_for_graph(0), {"a": 1, "b": 2})
        # Graph 1 matches only the second rule
        self.assertEqual(router.get_value_for_graph(1), {"b": 2})

    def test_config_router_conflict_raises(self, device):
        from torch._dynamo.graph_id_filter import GraphConfigRouter

        with self.assertRaisesRegex(ValueError, "Conflicting config override"):
            GraphConfigRouter("0:a=1;>=0:a=2")

    def test_config_router_same_value_no_conflict(self, device):
        from torch._dynamo.graph_id_filter import GraphConfigRouter

        router = GraphConfigRouter("0:a=1;>=0:a=1")
        self.assertEqual(router.get_value_for_graph(0), {"a": 1})
        self.assertEqual(router.get_value_for_graph(1), {"a": 1})

    def test_config_router_aggregation_multiple_rules(self, device):
        from torch._dynamo.graph_id_filter import GraphConfigRouter

        router = GraphConfigRouter("0:a=1;1:b=2;>=0:c=3")
        self.assertEqual(router.get_value_for_graph(0), {"a": 1, "c": 3})
        self.assertEqual(router.get_value_for_graph(1), {"b": 2, "c": 3})
        self.assertEqual(router.get_value_for_graph(2), {"c": 3})

    def test_backend_router_conflict_raises(self, device):
        from torch._dynamo.graph_id_filter import GraphBackendRouter

        with self.assertRaisesRegex(ValueError, "Conflicting backend override"):
            GraphBackendRouter("0-5:eager;3-10:inductor")

    def test_backend_router_same_backend_no_conflict(self, device):
        from torch._dynamo.graph_id_filter import GraphBackendRouter

        router = GraphBackendRouter("0:eager;>=0:eager")
        self.assertIsNotNone(router.get_value_for_graph(0))

    def test_get_inductor_config_override_empty(self, device):
        from torch._dynamo.graph_id_filter import (
            get_inductor_config_override_for_compile_id,
        )

        result = get_inductor_config_override_for_compile_id(None, "")
        self.assertIsNone(result)

    def test_combined_backend_and_config_override(self, device):
        """
        Test combining backend override with config override.

        Scenario: Default backend is eager, but override all graphs to use
        inductor with cudagraphs enabled, and additionally override graph 1
        to use cudagraph_skip_dynamic_graphs=False.
        """
        from torch._dynamo.graph_id_filter import (
            _create_backend_router,
            _create_inductor_config_router,
        )

        torch._dynamo.reset()
        _create_backend_router.cache_clear()
        _create_inductor_config_router.cache_clear()

        backends_used: list[str] = []
        configs_applied: list[dict] = []

        def fn(x):
            x = x + 1
            torch._dynamo.graph_break()
            x = x * 2
            torch._dynamo.graph_break()
            x = x - 1
            return x

        from torch._dynamo import output_graph

        original_wrap = output_graph._wrap_with_inductor_config

        def tracking_wrap(compiler_fn, config_patches):
            configs_applied.append(config_patches)
            return original_wrap(compiler_fn, config_patches)

        backend_override = ">=0:inductor"
        from torch._dynamo.graph_id_filter import get_backend_override_for_compile_id

        original_get_backend = get_backend_override_for_compile_id

        def tracking_get_backend(compile_id, config_str):
            result = original_get_backend(compile_id, config_str)
            if result is not None:
                backends_used.append("inductor")
            return result

        # Use both overrides:
        # - Backend: all graphs use inductor
        # - Config: all graphs enable cudagraphs, graph 1 also disables
        #   cudagraph_skip_dynamic_graphs
        with (
            patch.object(
                torch._dynamo.config,
                "debug_backend_override",
                backend_override,
            ),
            patch.object(
                torch._dynamo.config,
                "debug_inductor_config_override",
                "1:triton.cudagraph_skip_dynamic_graphs=False;>=0:triton.cudagraphs=True",
            ),
            patch(
                "torch._dynamo.output_graph.get_backend_override_for_compile_id",
                tracking_get_backend,
            ),
            patch.object(output_graph, "_wrap_with_inductor_config", tracking_wrap),
        ):
            compiled_fn = torch.compile(fn, backend="eager")
            compiled_fn(torch.randn(10, device=device))

        self.assertEqual(len(backends_used), 3)
        self.assertEqual(backends_used, ["inductor", "inductor", "inductor"])

        # All matching rules are aggregated. Graph 1 matches both rules.
        self.assertEqual(len(configs_applied), 3)
        self.assertEqual(configs_applied[0], {"triton.cudagraphs": True})
        self.assertEqual(
            configs_applied[1],
            {
                "triton.cudagraph_skip_dynamic_graphs": False,
                "triton.cudagraphs": True,
            },
        )
        self.assertEqual(configs_applied[2], {"triton.cudagraphs": True})

    def test_multiple_config_overrides_with_backend(self, device):
        """
        Test multiple config overrides applied to different graphs with backend override.

        Scenario: Default backend is eager, override graphs 0,2 to use inductor,
        and apply different config overrides to each.
        """
        from torch._dynamo.graph_id_filter import (
            _create_backend_router,
            _create_inductor_config_router,
        )

        torch._dynamo.reset()
        _create_backend_router.cache_clear()
        _create_inductor_config_router.cache_clear()

        backends_used: list[str] = []
        configs_applied: list[dict] = []

        def fn(x):
            x = x + 1
            torch._dynamo.graph_break()
            x = x * 2
            torch._dynamo.graph_break()
            x = x - 1
            torch._dynamo.graph_break()
            x = x / 2
            return x

        from torch._dynamo import output_graph

        original_wrap = output_graph._wrap_with_inductor_config

        def tracking_wrap(compiler_fn, config_patches):
            configs_applied.append(config_patches)
            return original_wrap(compiler_fn, config_patches)

        # Build backend tracking map
        backend_override = "0,2:inductor"
        backend_str_map: dict[int, str] = {}
        from torch._dynamo.graph_id_filter import GraphIdFilter

        for rule_str in backend_override.split(";"):
            if ":" not in rule_str:
                continue
            colon_idx = rule_str.find(":")
            filter_str = rule_str[:colon_idx].strip()
            backend_str = rule_str[colon_idx + 1 :].strip()
            gf = GraphIdFilter(filter_str)
            for gid in range(10):
                if gid in gf and gid not in backend_str_map:
                    backend_str_map[gid] = backend_str

        from torch._dynamo.graph_id_filter import get_backend_override_for_compile_id

        original_get_backend = get_backend_override_for_compile_id

        def tracking_get_backend(compile_id, config_str):
            result = original_get_backend(compile_id, config_str)
            if result is not None and compile_id.frame_id in backend_str_map:
                backends_used.append(backend_str_map[compile_id.frame_id])
            return result

        # Use both overrides:
        # - Backend: graphs 0,2 use inductor (graphs 1,3 stay with eager)
        # - Config: graph 0 disables cudagraphs, graph 2 disables skip_dynamic_graphs
        with (
            patch.object(
                torch._dynamo.config,
                "debug_backend_override",
                backend_override,
            ),
            patch.object(
                torch._dynamo.config,
                "debug_inductor_config_override",
                "0:triton.cudagraphs=False;2:triton.cudagraph_skip_dynamic_graphs=False",
            ),
            patch(
                "torch._dynamo.output_graph.get_backend_override_for_compile_id",
                tracking_get_backend,
            ),
            patch.object(output_graph, "_wrap_with_inductor_config", tracking_wrap),
        ):
            compiled_fn = torch.compile(fn, backend="eager")
            compiled_fn(torch.randn(10, device=device))

        self.assertEqual(len(backends_used), 2)
        self.assertEqual(backends_used, ["inductor", "inductor"])

        self.assertEqual(len(configs_applied), 2)
        self.assertIn({"triton.cudagraphs": False}, configs_applied)
        self.assertIn({"triton.cudagraph_skip_dynamic_graphs": False}, configs_applied)

    def test_config_override_backward_propagation(self, device):
        """
        Verify that inductor config overrides are active at inductor compile
        time for both forward and backward, across multiple graph breaks.
        """
        import torch._functorch.config
        from torch._dynamo.graph_id_filter import _create_inductor_config_router
        from torch._inductor import (
            compile_fx as compile_fx_mod,
            config as inductor_config,
        )

        torch._dynamo.reset()
        _create_inductor_config_router.cache_clear()

        TRACKED_CONFIGS = [
            "triton.cudagraphs",
            "triton.dense_indexing",
            "triton.cudagraph_skip_dynamic_graphs",
        ]

        def _read_config(key):
            obj = inductor_config
            for part in key.split("."):
                obj = getattr(obj, part)
            return obj

        baseline = {k: _read_config(k) for k in TRACKED_CONFIGS}
        configs_at_compile: dict[tuple[int, bool], dict] = {}

        original_compile_fx = compile_fx_mod.compile_fx
        original_inner_compile = compile_fx_mod.compile_fx_inner

        def tracking_inner_compile(gm, example_inputs, **kwargs):
            compile_id = torch._guards.CompileContext.current_compile_id()
            is_backward = kwargs.get("is_backward", False)
            snapshot = {k: _read_config(k) for k in TRACKED_CONFIGS}
            configs_at_compile[(compile_id.frame_id, is_backward)] = snapshot
            return original_inner_compile(gm, example_inputs, **kwargs)

        def tracking_compile_fx(model_, example_inputs_, *args, **kwargs):
            # Inject tracking inner_compile so compile_fx's config.patch
            # wrapping covers it for both forward and backward.
            if "inner_compile" not in kwargs:
                kwargs["inner_compile"] = tracking_inner_compile
            return original_compile_fx(model_, example_inputs_, *args, **kwargs)

        def fn(x):
            y = x * 2 + 1
            torch._dynamo.graph_break()
            z = y.sin()
            torch._dynamo.graph_break()
            return z.exp().sum()

        # Overlapping rules (all three configs default to False):
        config_override = (
            ">=1:triton.cudagraphs=True;"
            "0-1:triton.dense_indexing=True;"
            "1:triton.cudagraph_skip_dynamic_graphs=True"
        )
        expected_overrides = {
            0: {"triton.dense_indexing": True},
            1: {
                "triton.cudagraphs": True,
                "triton.dense_indexing": True,
                "triton.cudagraph_skip_dynamic_graphs": True,
            },
            2: {
                "triton.cudagraphs": True,
            },
        }

        with (
            patch.object(
                torch._dynamo.config,
                "debug_inductor_config_override",
                config_override,
            ),
            patch.object(compile_fx_mod, "compile_fx", tracking_compile_fx),
            patch.object(torch._functorch.config, "enable_autograd_cache", False),
        ):
            compiled_fn = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
            x = torch.randn(10, device=device, requires_grad=True)
            result = compiled_fn(x)
            result.backward()

        # Verify each graph has fwd+bwd, correct overrides, no cross-graph
        # leak, and identical configs for forward and backward.
        for gid in range(3):
            self.assertIn(
                (gid, False),
                configs_at_compile,
                lambda msg: f"{msg}\ngraph {gid} fwd missing",
            )
            self.assertIn(
                (gid, True),
                configs_at_compile,
                lambda msg: f"{msg}\ngraph {gid} bwd missing",
            )
            expected = {**baseline, **expected_overrides[gid]}
            for is_bw in [False, True]:
                phase = "backward" if is_bw else "forward"
                self.assertEqual(
                    configs_at_compile[(gid, is_bw)],
                    expected,
                    lambda msg: f"{msg}\ngraph {gid} {phase}: config mismatch",
                )

        self.assertIsNotNone(x.grad)


instantiate_device_type_tests(
    TestInductorConfigOverrideIntegration,
    globals(),
    allow_xpu=True,
)


class TestConfigOverrideValidation(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        from torch._dynamo.graph_id_filter import (
            _validate_backend_names,
            _validate_dynamo_config_keys,
            _validate_inductor_config_keys,
        )

        _validate_backend_names.cache_clear()
        _validate_dynamo_config_keys.cache_clear()
        _validate_inductor_config_keys.cache_clear()
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    @torch._dynamo.config.patch(
        debug_backend_override="0:not_a_real_backend",
    )
    def test_invalid_backend_raises_on_compile(self):
        def fn(x):
            return x + 1

        with self.assertRaisesRegex(ValueError, "not_a_real_backend"):
            torch.compile(fn, backend="eager")(torch.randn(4))

    @torch._dynamo.config.patch(
        debug_dynamo_config_override="0:nonexistent_dynamo_option=True",
    )
    def test_invalid_dynamo_config_raises_on_compile(self):
        def fn(x):
            return x + 1

        with self.assertRaisesRegex(ValueError, "nonexistent_dynamo_option"):
            torch.compile(fn, backend="eager")(torch.randn(4))

    @torch._dynamo.config.patch(
        debug_inductor_config_override="0:nonexistent_inductor_option=True",
    )
    def test_invalid_inductor_config_raises_on_compile(self):
        def fn(x):
            return x + 1

        with self.assertRaisesRegex(ValueError, "nonexistent_inductor_option"):
            torch.compile(fn, backend="eager")(torch.randn(4))


class TestDynamoConfigOverrideIntegration(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    @torch._dynamo.config.patch(
        specialize_float=False,
        verbose=False,
        debug_dynamo_config_override=(
            "0:specialize_float=True;1:verbose=True,recompile_limit=10"
        ),
    )
    def test_dynamo_config_override_per_graph(self):
        """Per-graph dynamo config overrides target the right graphs.

        Graph 0: specialize_float overridden True (base False)
        Graph 1: verbose+recompile_limit overridden (multiple keys)
        Graph 2: no override, keeps base values
        """
        from torch._dynamo.graph_id_filter import _create_dynamo_config_router

        _create_dynamo_config_router.cache_clear()

        observed: dict[int, dict] = {}

        def capturing_backend(gm, example_inputs):
            fid = torch._guards.CompileContext.current_compile_id().frame_id
            observed[fid] = {
                "specialize_float": torch._dynamo.config.specialize_float,
                "verbose": torch._dynamo.config.verbose,
                "recompile_limit": torch._dynamo.config.recompile_limit,
            }
            return gm

        def fn(x):
            x = x + 1
            torch._dynamo.graph_break()
            x = x * 2
            torch._dynamo.graph_break()
            return x - 1

        torch.compile(fn, backend=capturing_backend)(torch.randn(4))

        self.assertTrue(observed[0]["specialize_float"])
        self.assertFalse(observed[0]["verbose"])

        self.assertFalse(observed[1]["specialize_float"])
        self.assertTrue(observed[1]["verbose"])
        self.assertEqual(observed[1]["recompile_limit"], 10)

        self.assertFalse(observed[2]["specialize_float"])
        self.assertFalse(observed[2]["verbose"])

    def test_dynamo_config_override_warning(self):
        from torch._dynamo.graph_id_filter import _create_dynamo_config_router

        _create_dynamo_config_router.cache_clear()
        with self.assertWarnsRegex(
            UserWarning, "TORCH_COMPILE_OVERRIDE_DYNAMO_CONFIGS"
        ):
            _create_dynamo_config_router("0:specialize_float=True")


class TestStorageManifest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize("stable_hash", [None, False, True])
    def test_shared_store_requires_stable_hash(self, stable_hash):
        with TemporaryDirectoryName() as root:
            store = ContentStoreWriter(root)
            with self.assertRaisesRegex(ValueError, "stable_hash"):
                InputWriter(None, content_store=store, stable_hash=stable_hash)
            self.assertFalse(store.stable_hash)
            self.assertEqual(store.storage_metadata, {})
            store = ContentStoreWriter(root, stable_hash=True)
            if stable_hash is False:
                with self.assertRaisesRegex(ValueError, "stable_hash must match"):
                    InputWriter(None, content_store=store, stable_hash=stable_hash)
            else:
                writer = InputWriter(None, content_store=store, stable_hash=stable_hash)
                self.assertIs(writer.store, store)

    @parametrize("stable_hash", [None, False, True])
    def test_private_store_hash_setting_is_preserved(self, stable_hash):
        with TemporaryDirectoryName() as root:
            writer = InputWriter(root, stable_hash=stable_hash)
            self.assertEqual(writer.store.stable_hash, bool(stable_hash))

    @parametrize("field", ["logical_nbytes", "stored_nbytes"])
    @parametrize("value", [None, -1, 1.5, "4", True])
    def test_manifest_byte_sizes_must_be_nonnegative_integers(self, field, value):
        with TemporaryDirectoryName() as root:
            bundle, _, _ = self.write_bundle(root, [torch.ones(2)])
            path = os.path.join(bundle, "storage_manifest.json")
            with open(path) as f:
                manifest = json.load(f)
            (entry,) = manifest["storages"].values()
            if value is None:
                del entry[field]
            else:
                entry[field] = value
            with open(path, "w") as f:
                json.dump(manifest, f)
            with self.assertRaisesRegex(ValueError, f"'{field}' must be"):
                InputReader(bundle)

    @parametrize("manifest", [False, True])
    def test_loaded_size_must_match_generated_storage_size(self, manifest):
        with TemporaryDirectoryName() as root:
            if manifest:
                bundle, _, writer = self.write_bundle(root, [torch.ones(2)])
            else:
                bundle = root
                writer = InputWriter(root, stable_hash=True)
                writer.tensor("arg0", torch.ones(2))
            (h,) = writer.storage_hashes
            reader = InputReader(bundle)
            self.assertEqual(reader.storage(h, 8).nbytes(), 8)
            with self.assertRaisesRegex(ValueError, "expected 12, got 8"):
                reader.storage(h, 12)
            self.assertEqual(reader.args, [])

    @parametrize("compression", [None, "zstd"])
    def test_empty_storage_round_trip(self, compression):
        if compression == "zstd" and not zstd_available():
            self.skipTest("zstandard is not available")
        with TemporaryDirectoryName() as root:
            bundle, source, _ = self.write_bundle(
                root, [torch.empty(0)], compression=compression
            )
            self.assertEqual(self.load(bundle, source), [torch.empty(0)])

    def write_bundle(
        self, root: str, args: list[torch.Tensor], *, compression=None
    ) -> tuple[str, str, InputWriter]:
        bundle = os.path.join(root, "bundle")
        os.makedirs(bundle)
        store = ContentStoreWriter(
            os.path.join(root, "store"), stable_hash=True, compression=compression
        )
        writer = InputWriter(bundle, content_store=store)
        for i, arg in enumerate(args):
            writer.tensor(f"arg{i}", arg)
        writer.write_storage_manifest()
        return bundle, "\n".join(writer.lines()), writer

    def load(self, bundle: str, source: str) -> list[object]:
        namespace: dict[str, object] = {"torch": torch, "device": torch.device}
        exec(source, namespace)
        reader = InputReader(save_dir=bundle)
        namespace["load_args"](reader)
        return reader.args

    def test_round_trip_through_shared_store(self):
        x = torch.arange(8.0)
        args = [x, x[2:], torch.ones(3)]
        with TemporaryDirectoryName() as root:
            bundle, source, writer = self.write_bundle(root, args)
            with open(os.path.join(bundle, "storage_manifest.json")) as f:
                manifest = json.load(f)
            self.assertEqual(manifest["store"], os.path.join("..", "store"))
            self.assertEqual(set(manifest["storages"]), writer.storage_hashes)
            self.assertEqual(len(writer.storage_hashes), 2)
            loaded = self.load(bundle, source)
        self.assertEqual(loaded, args)

    @parametrize(
        "case,message",
        [
            ("not_object", "must be a JSON object"),
            ("schema_version", "unsupported storage manifest schema"),
            ("missing_store", "'store' must be a non-empty string"),
            ("empty_store", "'store' must be a non-empty string"),
            ("storages_list", "'storages' must be an object"),
            ("store_escapes", "escapes the capture root"),
            ("entry_list", "entries must be objects"),
            ("missing_path", "'path' must be a non-empty string"),
            ("empty_path", "'path' must be a non-empty string"),
            ("path_number", "'path' must be a non-empty string"),
            ("missing_codec", "unknown content-store codec"),
            ("unknown_codec", "unknown content-store codec"),
        ],
    )
    def test_malformed_manifest_is_rejected(self, case, message):
        with TemporaryDirectoryName() as root:
            bundle, _, _ = self.write_bundle(root, [torch.ones(2)])
            path = os.path.join(bundle, "storage_manifest.json")
            with open(path) as f:
                manifest = json.load(f)
            if case == "not_object":
                manifest = [manifest]
            elif case == "schema_version":
                manifest["schema_version"] = 2
            elif case == "missing_store":
                del manifest["store"]
            elif case == "empty_store":
                manifest["store"] = ""
            elif case == "storages_list":
                manifest["storages"] = []
            elif case == "store_escapes":
                manifest["store"] = os.path.join("..", "..", "elsewhere")
            else:
                (h,) = manifest["storages"]
                entry = manifest["storages"][h]
                if case == "entry_list":
                    manifest["storages"][h] = []
                elif case == "missing_path":
                    del entry["path"]
                elif case == "empty_path":
                    entry["path"] = ""
                elif case == "path_number":
                    entry["path"] = 1
                elif case == "missing_codec":
                    del entry["codec"]
                else:
                    entry["codec"] = "unknown"
            with open(path, "w") as f:
                json.dump(manifest, f)
            with self.assertRaisesRegex(ValueError, message):
                InputReader(save_dir=bundle)

    @parametrize("finalize", [False, True])
    def test_shared_store_requires_manifest(self, finalize):
        with TemporaryDirectoryName() as root:
            bundle = os.path.join(root, "bundle")
            writer = InputWriter(
                bundle,
                content_store=ContentStoreWriter(
                    os.path.join(root, "store"), stable_hash=True
                ),
            )
            writer.tensor("arg0", torch.ones(2))
            if finalize:
                writer.write_storage_manifest()
                os.unlink(os.path.join(bundle, "storage_manifest.json"))
            with self.assertRaisesRegex(FileNotFoundError, "storage manifest"):
                InputReader(save_dir=bundle)

    def test_required_marker_precedes_storage_and_preserves_metadata(self):
        with TemporaryDirectoryName() as root:
            bundle = os.path.join(root, "bundle")
            os.makedirs(bundle)
            path = os.path.join(bundle, "capture_metadata.json")
            with open(path, "w") as f:
                json.dump(
                    {"graph_name": "forward", "storage_manifest_required": False}, f
                )
            store = ContentStoreWriter(os.path.join(root, "store"), stable_hash=True)
            writer = InputWriter(bundle, content_store=store)
            write_storage = store.write_storage

            def checked_write(storage):
                with open(path) as f:
                    self.assertEqual(
                        json.load(f),
                        {"graph_name": "forward", "storage_manifest_required": True},
                    )
                return write_storage(storage)

            with patch.object(store, "write_storage", side_effect=checked_write):
                writer.tensor("arg0", torch.ones(2))
                writer.tensor("arg1", torch.zeros(3))
            writer.write_storage_manifest()
            with open(path) as f:
                self.assertEqual(
                    json.load(f),
                    {"graph_name": "forward", "storage_manifest_required": True},
                )

    def test_invalid_store_layout_fails_before_storage_write(self):
        with TemporaryDirectoryName() as root:
            bundle = os.path.join(root, "capture", "bundle")
            store = ContentStoreWriter(os.path.join(root, "outside"), stable_hash=True)
            writer = InputWriter(bundle, content_store=store)
            with patch.object(store, "write_storage") as write_storage:
                with self.assertRaisesRegex(ValueError, "escapes the capture root"):
                    writer.tensor("arg0", torch.ones(2))
                write_storage.assert_not_called()
            self.assertFalse(os.path.exists(bundle))
            with self.assertRaisesRegex(ValueError, "escapes the capture root"):
                writer.write_storage_manifest()

    @parametrize("metadata", [None, [], 1, "invalid"])
    def test_malformed_capture_metadata_is_rejected(self, metadata):
        with TemporaryDirectoryName() as root:
            bundle = os.path.join(root, "bundle")
            os.makedirs(bundle)
            with open(os.path.join(bundle, "capture_metadata.json"), "w") as f:
                json.dump(metadata, f)
            writer = InputWriter(
                bundle,
                content_store=ContentStoreWriter(
                    os.path.join(root, "store"), stable_hash=True
                ),
            )
            with self.assertRaisesRegex(ValueError, "capture metadata must be"):
                writer.tensor("arg0", torch.ones(2))
            with self.assertRaisesRegex(ValueError, "capture metadata must be"):
                InputReader(bundle)
            self.assertFalse(os.path.exists(os.path.join(root, "store", "storages")))

    @parametrize("compression", ["none", "zstd"])
    def test_two_bundles_deduplicate_and_replay_views(self, compression):
        if compression == "zstd" and not zstd_available():
            self.skipTest("zstandard is not available")
        x = torch.zeros(4096)
        args = [x, x[2::2]]
        with TemporaryDirectoryName() as root:
            store = ContentStoreWriter(
                os.path.join(root, "store"),
                stable_hash=True,
                compression="zstd" if compression == "zstd" else None,
            )
            for name in ("first", "second"):
                base = x.clone()
                args = [base, base[2::2]]
                writer = InputWriter(os.path.join(root, name), content_store=store)
                for i, arg in enumerate(args):
                    writer.tensor(f"arg{i}", arg)
                writer.write_storage_manifest()
                loaded = self.load(writer.save_dir, "\n".join(writer.lines()))
                self.assertEqual(loaded, args)
                self.assertEqual(loaded[1].stride(), args[1].stride())
                self.assertEqual(loaded[1].storage_offset(), args[1].storage_offset())
                self.assertEqual(
                    loaded[0].untyped_storage()._cdata,
                    loaded[1].untyped_storage()._cdata,
                )
                (h,) = writer.storage_hashes
                self.assertEqual(store.storage_metadata[h]["codec"], compression)
            storage_dir = os.path.join(root, "store", "storages")
            self.assertEqual(len(os.listdir(storage_dir)), 1)

    @parametrize("external_store", [False, True])
    def test_writer_without_save_dir_preserves_existing_behavior(self, external_store):
        x = torch.ones(4)
        with TemporaryDirectoryName() as root:
            store = (
                ContentStoreWriter(root, stable_hash=True) if external_store else None
            )
            writer = InputWriter(None, content_store=store)
            writer.tensor("arg0", x)
            namespace = {"torch": torch}
            exec("\n".join(writer.lines()), namespace)
            reader = InputReader(None)
            if store is not None:
                reader.store = ContentStoreReader(root)
            namespace["load_args"](reader)
            (loaded,) = reader.args
            if external_store:
                self.assertEqual(loaded, x)
            else:
                self.assertEqual(loaded.shape, x.shape)
            metadata_path = os.path.join(root, "capture_metadata.json")
            self.assertFalse(os.path.exists(metadata_path))

    def test_meta_storage_does_not_require_manifest(self):
        with TemporaryDirectoryName() as root:
            bundle = os.path.join(root, "bundle")
            writer = InputWriter(
                bundle,
                content_store=ContentStoreWriter(
                    os.path.join(root, "store"), stable_hash=True
                ),
            )
            writer.tensor("arg0", torch.empty(4, device="meta"))
            self.assertEqual(writer.storage_hashes, set())
            self.assertFalse(os.path.exists(bundle))

    @parametrize("marker", [None, False])
    def test_legacy_metadata_allows_random_fallback(self, marker):
        x = torch.ones(4)
        with TemporaryDirectoryName() as root:
            writer = InputWriter(root)
            writer.tensor("arg0", x)
            metadata = {} if marker is None else {"storage_manifest_required": marker}
            with open(os.path.join(root, "capture_metadata.json"), "w") as f:
                json.dump(metadata, f)
            (h,) = writer.storage_hashes
            os.unlink(os.path.join(root, "storages", h))
            (loaded,) = self.load(root, "\n".join(writer.lines()))
            self.assertEqual(loaded.shape, x.shape)

    def test_missing_storage_is_an_error_only_with_a_manifest(self):
        x = torch.ones(4)
        with TemporaryDirectoryName() as root:
            bundle, source, writer = self.write_bundle(root, [x])
            (h,) = writer.storage_hashes
            path = writer.store.storage_metadata[h]["path"]
            os.unlink(os.path.join(root, "store", path))
            with self.assertRaises(FileNotFoundError):
                self.load(bundle, source)

            legacy = os.path.join(root, "legacy")
            legacy_writer = InputWriter(legacy)
            legacy_writer.tensor("arg0", x)
            (legacy_h,) = legacy_writer.storage_hashes
            os.unlink(os.path.join(legacy, "storages", legacy_h))
            (loaded,) = self.load(legacy, "\n".join(legacy_writer.lines()))
        self.assertEqual(loaded.shape, x.shape)

    @parametrize("compression", [None, "zstd"])
    @parametrize("manifest", [False, True])
    def test_deduplicated_storages_stay_independent(self, compression, manifest):
        if compression == "zstd" and not zstd_available():
            self.skipTest("zstandard is not available")
        first, second = torch.zeros(4096), torch.zeros(4096)
        view = first[2::2]
        with TemporaryDirectoryName() as root:
            if manifest:
                bundle, source, writer = self.write_bundle(
                    root, [first, second, view], compression=compression
                )
            else:
                bundle = root
                writer = InputWriter(root, stable_hash=True)
                writer.store.compression = compression
                for i, arg in enumerate((first, second, view)):
                    writer.tensor(f"arg{i}", arg)
                source = "\n".join(writer.lines())
            self.assertEqual(len(writer.storage_hashes), 1)
            (h,) = writer.storage_hashes
            self.assertEqual(
                writer.store.storage_metadata[h]["codec"], compression or "none"
            )
            loaded_first, loaded_second, loaded_view = self.load(bundle, source)
        self.assertEqual(
            [loaded_first, loaded_second, loaded_view], [first, second, view]
        )
        first_ptr = loaded_first.untyped_storage().data_ptr()
        self.assertNotEqual(first_ptr, loaded_second.untyped_storage().data_ptr())
        self.assertEqual(first_ptr, loaded_view.untyped_storage().data_ptr())
        self.assertEqual(loaded_view.storage_offset(), 2)
        self.assertEqual(loaded_view.stride(), view.stride())
        loaded_view.add_(3)
        expected = torch.zeros_like(first)
        expected[2::2] = 3
        self.assertEqual(loaded_first, expected)
        self.assertEqual(loaded_second, second)
        self.assertEqual(loaded_view, torch.full_like(view, 3))

    def test_load_args_without_inputs(self):
        writer = InputWriter(None)
        namespace: dict[str, object] = {}
        exec("\n".join(writer.lines()), namespace)
        reader = InputReader()
        namespace["load_args"](reader)
        self.assertEqual(reader.args, [])
        with self.assertRaisesRegex(RuntimeError, "needs a save_dir"):
            writer.write_storage_manifest()


instantiate_parametrized_tests(TestStorageManifest)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
