# Owner(s): ["module: onnx"]
from __future__ import annotations

import logging

import tempfile

from typing import Mapping, Tuple

import onnx
import onnx.inliner
import pytorch_test_common
import transformers  # type: ignore[import]

import torch
from torch import nn
from torch._subclasses import fake_tensor
from torch.nn import functional as F
from torch.onnx import dynamo_export, ExportOptions
from torch.onnx._internal.diagnostics import infra  # noqa: TCH001
from torch.onnx._internal.fx import diagnostics, registration
from torch.testing._internal import common_utils


def assert_has_diagnostics(
    diagnostic_context: diagnostics.DiagnosticContext,
    rule: infra.Rule,
    level: infra.Level,
    expected_node: str,
):
    rule_level_pairs = (rule.id, level.name.lower())
    sarif_log = diagnostic_context.sarif_log()
    actual_results = []
    for run in sarif_log.runs:
        if run.results is None:
            continue
        for result in run.results:
            id_level_pair = (result.rule_id, result.level)
            actual_results.append(id_level_pair)
            if (
                rule_level_pairs == id_level_pair
                and result.message.text
                and result.message.markdown
                and expected_node in result.message.text
            ):
                return

    raise AssertionError(
        f"Expected diagnostic results of rule id and level pair {rule_level_pairs} "
        f"not found with expected error node {expected_node} and "
        f"Actual diagnostic results: {actual_results}"
    )


@common_utils.instantiate_parametrized_tests
class TestFxToOnnx(pytorch_test_common.ExportTestCase):
    def setUp(self):
        super().setUp()
        self.export_options = ExportOptions()

    def tearDown(self):
        super().tearDown()

    def test_simple_function(self):
        def func(x):
            y = x + 1
            z = y.relu()
            return (y, z)

        _ = dynamo_export(
            func, torch.randn(1, 1, 2), export_options=self.export_options
        )

    def test_empty(self):
        # Since `torch.empty` returns tensor with uninitialized data, we cannot
        # test this under `test_fx_to_onnx_with_onnxruntime.py` with result comparison.
        def func(x):
            return torch.empty(x.size(), dtype=torch.int64)

        tensor_x = torch.randn(1, 1, 2)
        _ = dynamo_export(func, tensor_x, export_options=self.export_options)

    def test_args_used_for_export_is_not_converted_to_fake_tensors(self):
        def func(x, y):
            return x + y

        tensor_x = torch.randn(1, 1, 2)
        tensor_y = torch.randn(1, 1, 2)
        _ = dynamo_export(func, tensor_x, tensor_y, export_options=self.export_options)
        self.assertNotIsInstance(tensor_x, fake_tensor.FakeTensor)
        self.assertNotIsInstance(tensor_y, fake_tensor.FakeTensor)

    @common_utils.parametrize(
        "diagnostic_rule",
        [
            common_utils.subtest(
                diagnostics.rules.find_opschema_matched_symbolic_function,
                name="optional_inputs",
            ),
            common_utils.subtest(
                diagnostics.rules.op_level_debugging,
                name="get_attr_node_in_op_level_debug",
            ),
        ],
    )
    def test_mnist_exported_with_no_warnings(self, diagnostic_rule):
        class MNISTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
                self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
                self.fc1 = nn.Linear(9216, 128, bias=False)
                self.fc2 = nn.Linear(128, 10, bias=False)

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.conv1(tensor_x)
                tensor_x = F.sigmoid(tensor_x)
                tensor_x = self.conv2(tensor_x)
                tensor_x = F.sigmoid(tensor_x)
                tensor_x = F.max_pool2d(tensor_x, 2)
                tensor_x = torch.flatten(tensor_x, 1)
                tensor_x = self.fc1(tensor_x)
                tensor_x = F.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                output = F.log_softmax(tensor_x, dim=1)
                return output

        tensor_x = torch.rand((64, 1, 28, 28), dtype=torch.float32)
        onnx_program = dynamo_export(
            MNISTModel(), tensor_x, export_options=ExportOptions(op_level_debug=True)
        )

        assert_has_diagnostics(
            onnx_program.diagnostic_context,
            diagnostic_rule,
            diagnostics.levels.NONE,
            expected_node="aten.convolution.default",
        )

    def test_no_warnings_on_complex_dtype_in_op_level_debug(self):
        class ComplexModel(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.mul(input, input)

        real = torch.tensor([1, 2], dtype=torch.float32)
        imag = torch.tensor([3, 4], dtype=torch.float32)
        x = torch.complex(real, imag)

        onnx_program = dynamo_export(
            ComplexModel(), x, export_options=ExportOptions(op_level_debug=True)
        )

        assert_has_diagnostics(
            onnx_program.diagnostic_context,
            diagnostics.rules.op_level_debugging,
            diagnostics.levels.NONE,
            expected_node="aten.mul.Tensor",
        )

    def test_trace_only_op_with_evaluator(self):
        model_input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 2.0]])

        class ArgminArgmaxModel(torch.nn.Module):
            def forward(self, input):
                return (
                    torch.argmin(input),
                    torch.argmax(input),
                    torch.argmin(input, keepdim=True),
                    torch.argmax(input, keepdim=True),
                    torch.argmin(input, dim=0, keepdim=True),
                    torch.argmax(input, dim=1, keepdim=True),
                )

        # NOTE: KeyError: dim raised in optimizer
        with self.assertWarnsOnceRegex(
            UserWarning, "ONNXScript optimizer failed. Skipping optimization."
        ):
            _ = dynamo_export(
                ArgminArgmaxModel(), model_input, export_options=self.export_options
            )

    def test_multiple_outputs_op_with_evaluator(self):
        class TopKModel(torch.nn.Module):
            def forward(self, x):
                values, _ = torch.topk(x, 3)
                return torch.sum(values)

        x = torch.arange(1.0, 6.0, requires_grad=True)

        _ = dynamo_export(TopKModel(), x, export_options=self.export_options)

    def test_unsupported_indices_fake_tensor_generated_with_op_level_debug(self):
        class EmbedModelWithoutPaddingIdx(torch.nn.Module):
            def forward(self, input, emb):
                return torch.nn.functional.embedding(input, emb)

        model = EmbedModelWithoutPaddingIdx()
        x = torch.randint(4, (4, 3, 2))
        embedding_matrix = torch.rand(10, 3)

        onnx_program = dynamo_export(
            model,
            x,
            embedding_matrix,
            export_options=ExportOptions(op_level_debug=True),
        )
        assert_has_diagnostics(
            onnx_program.diagnostic_context,
            diagnostics.rules.op_level_debugging,
            diagnostics.levels.WARNING,
            expected_node="aten.embedding.default",
        )

    def test_unsupported_function_schema_raises_diagnostic_warning_when_found_nearest_match(
        self,
    ):
        class TraceModel(torch.nn.Module):
            def forward(self, input):
                return input.new_zeros(())

        x = torch.randn((2, 3), dtype=torch.float32)
        onnx_program = dynamo_export(TraceModel(), x)

        assert_has_diagnostics(
            onnx_program.diagnostic_context,
            diagnostics.rules.find_opschema_matched_symbolic_function,
            diagnostics.levels.WARNING,
            expected_node="aten.new_zeros.default",
        )

    def test_perfect_match_on_sequence_and_bool_attributes(
        self,
    ):
        class TraceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2 = torch.nn.Conv2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )

            def forward(self, input):
                return self.conv2(input)

        x = torch.randn(20, 16, 50, 50)
        onnx_program = dynamo_export(
            TraceModel(), x, export_options=ExportOptions(op_level_debug=False)
        )
        assert_has_diagnostics(
            onnx_program.diagnostic_context,
            diagnostics.rules.find_opschema_matched_symbolic_function,
            diagnostics.levels.NONE,
            expected_node="aten.convolution.default",
        )

    def test_dispatch_overload_fall_back_default_raise_diagnostic_warning(self):
        class TraceModel(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.add.Tensor(input, input)

        onnx_registry = torch.onnx.OnnxRegistry()
        self.assertTrue(
            onnx_registry.is_registered_op(
                namespace="aten", op_name="add", overload="Tensor"
            )
        )

        aten_add_Tensor = registration.OpName.from_name_parts(
            namespace="aten", op_name="add", overload="Tensor"
        )
        onnx_registry._registry.pop(aten_add_Tensor)

        x = torch.tensor(3)
        onnx_program = dynamo_export(
            TraceModel(), x, export_options=ExportOptions(onnx_registry=onnx_registry)
        )
        assert_has_diagnostics(
            onnx_program.diagnostic_context,
            diagnostics.rules.find_operator_overloads_in_onnx_registry,
            diagnostics.levels.WARNING,
            expected_node="aten.add.Tensor",
        )

    def test_aten_clone_does_not_raise_warning_of_lack_of_memory_format(self):
        class CustomModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.clone(input, memory_format=torch.preserve_format)

        x = torch.tensor(3)
        onnx_program = dynamo_export(CustomModule(), x)
        assert_has_diagnostics(
            onnx_program.diagnostic_context,
            diagnostics.rules.find_opschema_matched_symbolic_function,
            diagnostics.levels.NONE,
            expected_node="aten.clone.default",
        )

    def test_missing_complex_onnx_variant_raises_errors_in_dispatcher(self):
        registry = torch.onnx.OnnxRegistry()

        # NOTE: simulate unsupported nodes
        aten_mul_tensor = registration.OpName.from_name_parts(
            namespace="aten", op_name="mul", overload="Tensor"
        )

        # Only keep real aten.mul to test missing complex aten.mul
        registry._registry[aten_mul_tensor] = [
            onnx_func
            for onnx_func in registry._registry[aten_mul_tensor]
            if not onnx_func.is_complex
        ]

        class TraceModel(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.mul.Tensor(input, input)

        x = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)

        with self.assertRaises(torch.onnx.OnnxExporterError) as e:
            torch.onnx.dynamo_export(
                TraceModel(),
                x,
                export_options=torch.onnx.ExportOptions(onnx_registry=registry),
            )

        try:
            torch.onnx.dynamo_export(
                TraceModel(),
                x,
                export_options=torch.onnx.ExportOptions(onnx_registry=registry),
            )
        except torch.onnx.OnnxExporterError as e:
            assert_has_diagnostics(
                e.onnx_program.diagnostic_context,
                diagnostics.rules.no_symbolic_function_for_call_function,
                diagnostics.levels.ERROR,
                expected_node="aten.mul.Tensor",
            )

    def test_symbolic_shape_of_values_inside_function_is_exported_as_graph_value_info(
        self,
    ):
        class SubModule(torch.nn.Module):
            def forward(self, x, y, bias):
                output = x @ y
                return output + bias

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submodule = SubModule()

            def forward(self, x, y, bias):
                return self.submodule(x, y, bias)

        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        bias = torch.randn(4)
        onnx_program = torch.onnx.dynamo_export(
            Module(),
            x,
            y,
            bias,
            export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
        )
        model_proto = onnx_program.model_proto

        # Assert value_info for values inside local function can be retrieved
        def _assert_node_outputs_has_value_info(
            node: onnx.NodeProto,
            value_infos: Mapping[str, onnx.ValueInfoProto],
            local_functions: Mapping[Tuple[str, str], onnx.FunctionProto],
            exclude_names_in_value_info,
            function_id: str = "",
        ):
            for output in node.output:
                name = f"{function_id}/{output}" if function_id else output
                if name not in exclude_names_in_value_info:
                    self.assertIn(name, value_infos)
            if node.domain.startswith("pkg.onnxscript.torch_lib"):
                # No shape info available for values inside torchlib functions.
                return
            if (
                function := local_functions.get((node.domain, node.op_type))
            ) is not None:
                for node in function.node:
                    function_id = f"{function.domain}::{function.name}"
                    _assert_node_outputs_has_value_info(
                        node,
                        value_infos,
                        local_functions,
                        exclude_names_in_value_info,
                        function_id,
                    )

        type_infos = {vi.name: vi for vi in model_proto.graph.value_info}
        functions = {(f.domain, f.name): f for f in model_proto.functions}
        # NOTE: inputs, outputs, and initializers are not included in value_info spec
        exclude_names_in_value_info = (
            [input.name for input in model_proto.graph.input]
            + [output.name for output in model_proto.graph.output]
            + [init.name for init in model_proto.graph.initializer]
        )
        for node in model_proto.graph.node:
            _assert_node_outputs_has_value_info(
                node, type_infos, functions, exclude_names_in_value_info
            )

    def test_dynamo_export_retains_readable_parameter_and_buffer_names(self):
        class SubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
                self.fc1 = nn.Linear(9216, 128, bias=False)
                self.register_buffer("buffer", torch.randn(1, 128))

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.conv2(tensor_x)
                tensor_x = F.sigmoid(tensor_x)
                tensor_x = F.max_pool2d(tensor_x, 2)
                tensor_x = torch.flatten(tensor_x, 1)
                tensor_x = self.fc1(tensor_x)
                tensor_x = tensor_x + self.buffer
                tensor_x = F.sigmoid(tensor_x)
                return tensor_x

        class MNISTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
                self.submodule = SubModule()
                self.fc2 = nn.Linear(128, 10, bias=False)

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.conv1(tensor_x)
                tensor_x = F.sigmoid(tensor_x)
                tensor_x = self.submodule(tensor_x)
                tensor_x = self.fc2(tensor_x)
                output = F.log_softmax(tensor_x, dim=1)
                return output

        tensor_x = torch.rand((64, 1, 28, 28), dtype=torch.float32)

        model = MNISTModel()
        onnx_program = torch.onnx.dynamo_export(model, tensor_x)
        model_proto = onnx_program.model_proto

        # NOTE: initializers could be optimized away by onnx optimizer
        onnx_initilizers = {init.name for init in model_proto.graph.initializer}
        torch_weights = {*model.state_dict().keys()}
        self.assertTrue(onnx_initilizers.issubset(torch_weights))

    @common_utils.parametrize(
        "checkpoint_type",
        [
            common_utils.subtest(
                "state_dict",
                name="state_dict",
            ),
            common_utils.subtest(
                "state_dict",
                name="checkpoint_file",
            ),
        ],
    )
    def test_fake_tensor_mode_simple(self, checkpoint_type):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                out = self.linear(x)
                return out

        with torch.onnx.enable_fake_mode() as fake_context:
            x = torch.rand(5, 2, 2)
            model = Model()
            export_options = ExportOptions(fake_context=fake_context)
            onnx_program = torch.onnx.dynamo_export(
                model, x, export_options=export_options
            )

        assert (
            onnx_program is not None
        ), "ONNXProgram must be created on successful export"
        assert (
            onnx_program.model_proto is not None
        ), "A model protobuf must be created on a successful export"
        onnx.checker.check_model(onnx_program.model_proto, full_check=True)
        assert (
            len(onnx_program.model_proto.graph.initializer) == 0
        ), "Initializers cannot exist when fake mode is enabled"

        if checkpoint_type == "state_dict":
            # Variant 1: Save ONNX proto using Model's state_dict()
            with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp_onnx_file:
                model_state_dict = (
                    Model().state_dict()
                )  # Create a state_dict for testing
                onnx_program.save(tmp_onnx_file.name, model_state=model_state_dict)
                assert (
                    len(onnx.load(tmp_onnx_file.name).graph.initializer) == 2
                ), "Initializers must be present after loading it from model_state_dict"
                # Let's make sure consecutive `save` calls don't create dupes
                onnx_program.save(tmp_onnx_file.name, model_state=model_state_dict)
                assert (
                    len(onnx.load(tmp_onnx_file.name).graph.initializer) == 2
                ), "Initializers must be present after loading it from model_state_dict"
        elif checkpoint_type == "checkpoint_file":
            # Variant 2: Save ONNX proto using Model checkpoint file
            with tempfile.NamedTemporaryFile(
                suffix=".onnx"
            ) as tmp_onnx_file, tempfile.NamedTemporaryFile(
                suffix=".pt"
            ) as tmp_checkpoint_file:
                torch.save(
                    Model().state_dict(), tmp_checkpoint_file.name
                )  # Create checkpoint file for testing
                onnx_program.save(
                    tmp_onnx_file.name, model_state=tmp_checkpoint_file.name
                )
                assert (
                    len(onnx.load(tmp_onnx_file.name).graph.initializer) == 2
                ), "Initializers must be present after loading it from model_state_dict"
                # Let's make sure consecutive `save` calls don't create dupes
                onnx_program.save(
                    tmp_onnx_file.name, model_state=tmp_checkpoint_file.name
                )
                assert (
                    len(onnx.load(tmp_onnx_file.name).graph.initializer) == 2
                ), "Initializers must be present after loading it from model_state_dict"

    def test_fake_tensor_mode_simple_invalid_input(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                out = self.linear(x)
                return out

        real_model = Model()
        real_x = torch.rand(5, 2, 2)
        with torch.onnx.enable_fake_mode() as fake_context:
            fake_model = Model()
            fake_x = torch.rand(5, 2, 2)

            # TODO: Split each scenario on its own test case
            # Scenario 1: Fake model and fake input WITHOUT ExportOptions(fake_context=...)
            with self.assertRaises(torch.onnx.OnnxExporterError):
                export_options = ExportOptions(fake_context=None)
                _ = torch.onnx.dynamo_export(
                    fake_model, fake_x, export_options=export_options
                )

            # Scenario 2: Fake model and real input WITHOUT fake_context
            with self.assertRaises(torch.onnx.OnnxExporterError):
                export_options = ExportOptions(fake_context=None)
                _ = torch.onnx.dynamo_export(
                    fake_model, real_x, export_options=export_options
                )

            # Scenario 3: Real model and real input WITH fake_context
            with self.assertRaises(torch.onnx.OnnxExporterError):
                export_options = ExportOptions(fake_context=fake_context)
                _ = torch.onnx.dynamo_export(
                    real_model, real_x, export_options=export_options
                )

            # Scenario 4: Fake model and real input WITH fake_context
            with self.assertRaises(torch.onnx.OnnxExporterError):
                export_options = ExportOptions(fake_context=fake_context)
                _ = torch.onnx.dynamo_export(
                    fake_model, real_x, export_options=export_options
                )

    @pytorch_test_common.xfail(
        error_message="Dynamic control flow is not supported at the moment."
    )
    def test_fake_tensor_mode_huggingface_llama(self):
        config = transformers.LlamaConfig(
            vocab_size=8096, hidden_size=256, num_hidden_layers=2, num_attention_heads=2
        )
        batch, seq = 4, 256

        with torch.onnx.enable_fake_mode() as fake_context:
            model = transformers.LlamaModel(config).eval()
            input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            attention_mask = torch.ones(batch, seq, dtype=torch.bool)
            position_ids = torch.arange(0, seq, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).view(-1, seq)

            export_options = torch.onnx.ExportOptions(fake_context=fake_context)
            onnx_program = torch.onnx.dynamo_export(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                export_options=export_options,
            )
            onnx.checker.check_model(onnx_program.model_proto)
            onnx.shape_inference.infer_shapes(onnx_program.model_proto)

    @pytorch_test_common.xfail(
        error_message="Dynamic control flow is not supported at the moment."
    )
    def test_fake_tensor_mode_huggingface_tiiuae_falcon(self):
        config = transformers.FalconConfig()
        batch, seq = 4, 256

        with torch.onnx.enable_fake_mode() as fake_context:
            model = transformers.FalconModel(config).eval()
            input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            attention_mask = torch.ones(batch, seq, dtype=torch.bool)

            export_options = torch.onnx.ExportOptions(fake_context=fake_context)
            onnx_program = torch.onnx.dynamo_export(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                export_options=export_options,
            )
            onnx.checker.check_model(onnx_program.model_proto)
            onnx.shape_inference.infer_shapes(onnx_program.model_proto)

    def test_exported_program_input_with_custom_fx_tracer(self):
        from torch.onnx._internal import exporter
        from torch.onnx._internal.fx import dynamo_graph_extractor

        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1

        x = torch.randn(1, 1, 2)
        exported_program = torch.export.export(Model(), args=(x,))

        export_options = torch.onnx.ExportOptions()
        export_options = exporter.ResolvedExportOptions(
            export_options, model=exported_program
        )
        export_options.fx_tracer = (
            dynamo_graph_extractor.DynamoExport()
        )  # Override fx_tracer to an unsupported tracer
        with self.assertRaises(torch.onnx.OnnxExporterError):
            onnx_program = torch.onnx.dynamo_export(
                exported_program,
                x,
                export_options=export_options,
            )
            self.assertTrue(onnx_program._export_exception is not None)
            with self.assertRaises(torch.onnx.InvalidExportOptionsError):
                raise self._export_exception

    def test_exported_program_torch_distributions_normal_Normal(self):
        class Model(torch.nn.Module):
            def __init__(self):
                self.normal = torch.distributions.normal.Normal(0, 1)
                super().__init__()

            def forward(self, x):
                return self.normal.sample(x.shape)

        x = torch.randn(2, 3)
        with torch.no_grad():
            exported_program = torch.export.export(Model(), args=(x,))
            _ = torch.onnx.dynamo_export(
                exported_program,
                x,
            )

    def test_aten_div_no_opmath_type_promotion(self):
        class Model(torch.nn.Module):
            def forward(self, input):
                return input / 2

        model = Model()
        input = torch.randn(3, 5, requires_grad=True, dtype=torch.float16)

        model_proto = torch.onnx.dynamo_export(model, input).model_proto
        model_proto = onnx.inliner.inline_local_functions(model_proto)
        div_node = next(
            node for node in model_proto.graph.node if node.op_type == "Div"
        )
        # The input of Div node should be the input of the model,
        # with no Cast node in between.
        self.assertEqual(div_node.input[0], model_proto.graph.input[0].name)

    def test_exported_program_as_input_with_model_signature(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1.0

        x = torch.randn(1, 1, 2, dtype=torch.float)
        exported_program = torch.export.export(Model(), args=(x,))

        onnx_program = torch.onnx.dynamo_export(
            exported_program,
            x,
        )

        self.assertTrue(onnx_program.model_signature, torch.export.ExportGraphSignature)

    @common_utils.parametrize(
        "float8_type",
        [
            common_utils.subtest(
                torch.float8_e5m2,
                name="torch_float8_e5m2",
            ),
            common_utils.subtest(
                torch.float8_e5m2fnuz,
                name="torch_float8_e5m2fnuz",
            ),
            common_utils.subtest(
                torch.float8_e4m3fn,
                name="torch_float8_e4m3fn",
            ),
            common_utils.subtest(
                torch.float8_e4m3fnuz,
                name="torch_float8_e4m3fnuz",
            ),
        ],
    )
    def test_float8_support(self, float8_type):
        class Float8Module(torch.nn.Module):
            def forward(self, input: torch.Tensor):
                input = input.to(float8_type)
                return input + torch.tensor(1.0, dtype=float8_type)

        # NOTE: shape inference error raised in optimizer due to unsupported dtype
        with self.assertWarnsOnceRegex(
            UserWarning, "ONNXScript optimizer failed. Skipping optimization."
        ):
            _ = torch.onnx.dynamo_export(Float8Module(), torch.randn(1, 2, 3, 4))

    def test_export_with_logging_logger(self):
        logger = logging.getLogger(__name__)

        class LoggingLoggerModule(torch.nn.Module):
            def forward(self, x):
                logger.log("abc")
                return x + 1

        input = torch.randn(2, 3)
        model = LoggingLoggerModule()
        _ = torch.onnx.dynamo_export(model, input)

    def test_export_with_hf_logging_logger(self):
        logger = transformers.utils.logging.get_logger(__name__)

        class HFLoggingLoggerModule(torch.nn.Module):
            def forward(self, x):
                logger.warning_once("abc")
                return x + 1

        input = torch.randn(2, 3)
        model = HFLoggingLoggerModule()
        _ = torch.onnx.dynamo_export(model, input)

    def test_checkpoint_cast(self):
        model_id = "openai/whisper-large-v3"
        feature_extractor = transformers.WhisperFeatureExtractor(feature_size=128)
        batch = 4

        with torch.onnx.enable_fake_mode() as ctx:
            model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, low_cpu_mem_usage=False, use_safetensors=False
            )
            input = {
                "input_features": torch.randn(
                    (
                        batch,
                        feature_extractor.feature_size,
                        feature_extractor.nb_max_frames,
                    )
                ),
                "decoder_input_ids": torch.tensor([[1, 1]]) * 8001,
                "return_dict": False,
            }

        export_options = torch.onnx.ExportOptions(fake_context=ctx)
        onnx_program = torch.onnx.dynamo_export(
            model, **input, export_options=export_options
        )
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp_onnx_file:
            onnx_program.save(tmp_onnx_file.name)
            onnx.checker.check_model(tmp_onnx_file.name, full_check=True)

    @common_utils.parametrize(
        "include_initializer",
        [
            common_utils.subtest(
                True,
                name="include_initializer",
            ),
            common_utils.subtest(
                False,
                name="dont_include_initializer",
            ),
        ],
    )
    @common_utils.parametrize(
        "use_fake_mode",
        [
            common_utils.subtest(
                True,
                name="use_fake_mode",
            ),
            common_utils.subtest(
                False,
                name="no_fake_mode",
            ),
        ],
    )
    @common_utils.parametrize(
        "use_exported_program",
        [
            common_utils.subtest(
                True,
                name="use_exported_program",
            ),
            common_utils.subtest(
                False,
                name="no_exported_program",
            ),
        ],
    )
    def test_save_with_without_initializer(
        self, include_initializer, use_fake_mode, use_exported_program
    ):
        class MNISTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
                self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
                self.fc1 = nn.Linear(9216, 128, bias=False)
                self.fc2 = nn.Linear(128, 10, bias=False)

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.conv1(tensor_x)
                tensor_x = F.sigmoid(tensor_x)
                tensor_x = self.conv2(tensor_x)
                tensor_x = F.sigmoid(tensor_x)
                tensor_x = F.max_pool2d(tensor_x, 2)
                tensor_x = torch.flatten(tensor_x, 1)
                tensor_x = self.fc1(tensor_x)
                tensor_x = F.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                output = F.log_softmax(tensor_x, dim=1)
                return output

        state_dict = MNISTModel().state_dict()
        if use_fake_mode:
            with torch.onnx.enable_fake_mode() as ctx:
                model = MNISTModel()
                tensor_x = torch.rand((64, 1, 28, 28), dtype=torch.float32)
                if use_exported_program:
                    model = torch.export.export(model, args=(tensor_x,))
                export_options = torch.onnx.ExportOptions(fake_context=ctx)
        else:
            model = MNISTModel()
            tensor_x = torch.rand((64, 1, 28, 28), dtype=torch.float32)
            if use_exported_program:
                model = torch.export.export(model, args=(tensor_x,))
            export_options = torch.onnx.ExportOptions()

        onnx_program = torch.onnx.dynamo_export(
            model, tensor_x, export_options=export_options
        )
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp_onnx_file:
            onnx_program.save(
                tmp_onnx_file.name,
                include_initializers=include_initializer,
                model_state=state_dict if include_initializer else None,
            )
            onnx_model = onnx.load(tmp_onnx_file.name)
            self.assertEqual(
                (include_initializer and len(onnx_model.graph.initializer) > 0)
                or (not include_initializer and len(onnx_model.graph.initializer) == 0),
                True,
            )

    def test_export_with_print(self):
        class PrintModule(torch.nn.Module):
            def forward(self, x):
                print("abc")
                return x + 1

        input = torch.randn(2, 3)
        model = PrintModule()
        _ = torch.onnx.dynamo_export(model, input)


if __name__ == "__main__":
    common_utils.run_tests()
