# Owner(s): ["module: onnx"]
from __future__ import annotations

import tempfile

import onnx
import pytorch_test_common
import torch
import transformers  # type: ignore[import]
from torch import nn
from torch._subclasses import fake_tensor
from torch.nn import functional as F
from torch.onnx import dynamo_export, ExportOptions
from torch.onnx._internal.diagnostics import infra
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
        export_output = dynamo_export(
            MNISTModel(), tensor_x, export_options=ExportOptions(op_level_debug=True)
        )

        assert_has_diagnostics(
            export_output.diagnostic_context,
            diagnostic_rule,
            diagnostics.levels.NONE,
            expected_node="aten.convolution.default",
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

        _ = dynamo_export(
            ArgminArgmaxModel(), model_input, export_options=self.export_options
        )

    def test_multiple_outputs_op_with_evaluator(self):
        class TopKModel(torch.nn.Module):
            def forward(self, x):
                values, _ = torch.topk(x, 3)
                return torch.sum(values)

        x = torch.arange(1.0, 6.0, requires_grad=True)
        export_output = dynamo_export(
            TopKModel(), x, export_options=self.export_options
        )

    def test_unsupported_indices_fake_tensor_generated_with_op_level_debug(self):
        class EmbedModelWithoutPaddingIdx(torch.nn.Module):
            def forward(self, input, emb):
                return torch.nn.functional.embedding(input, emb)

        model = EmbedModelWithoutPaddingIdx()
        x = torch.randint(4, (4, 3, 2))
        embedding_matrix = torch.rand(10, 3)

        export_output = dynamo_export(
            model,
            x,
            embedding_matrix,
            export_options=ExportOptions(op_level_debug=True),
        )
        assert_has_diagnostics(
            export_output.diagnostic_context,
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
        export_output = dynamo_export(TraceModel(), x)

        assert_has_diagnostics(
            export_output.diagnostic_context,
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
        export_output = dynamo_export(
            TraceModel(), x, export_options=ExportOptions(op_level_debug=False)
        )
        assert_has_diagnostics(
            export_output.diagnostic_context,
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
        # TODO: Replace this example with a torch custom op when overload is supported
        # Currently, torch only supports custom op with namespace and op_name
        aten_add_Tensor = registration.OpName.from_name_parts(
            namespace="aten", op_name="add", overload="Tensor"
        )
        onnx_registry._registry.pop(aten_add_Tensor)

        x = torch.tensor(3)
        export_output = dynamo_export(
            TraceModel(), x, export_options=ExportOptions(onnx_registry=onnx_registry)
        )
        assert_has_diagnostics(
            export_output.diagnostic_context,
            diagnostics.rules.find_operator_overloads_in_onnx_registry,
            diagnostics.levels.WARNING,
            expected_node="aten.add.Tensor",
        )

    def test_aten_clone_does_not_raise_warning_of_lack_of_memory_format(self):
        class CustomModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.clone(input, memory_format=torch.preserve_format)

        x = torch.tensor(3)
        export_output = dynamo_export(CustomModule(), x)
        assert_has_diagnostics(
            export_output.diagnostic_context,
            diagnostics.rules.find_opschema_matched_symbolic_function,
            diagnostics.levels.NONE,
            expected_node="aten.clone.default",
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
        export_output = torch.onnx.dynamo_export(model, tensor_x)
        model_proto = export_output.model_proto
        self.assertEqual(
            {initializer.name for initializer in model_proto.graph.initializer},
            {*model.state_dict().keys()},
        )

    def test_fake_tensor_mode_simple(self):
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
            export_output = torch.onnx.dynamo_export(
                model, x, export_options=export_options
            )

        assert (
            export_output is not None
        ), "ExportOutput must be created on successful export"
        assert (
            export_output.model_proto is not None
        ), "A model protobuf must be created on a successful export"
        onnx.checker.check_model(export_output.model_proto, full_check=True)
        assert (
            len(export_output.model_proto.graph.initializer) == 0
        ), "Initializers cannot exist when fake mode is enabled"

        # Variant 1: Save ONNX proto using Model's state_dict()
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp_onnx_file:
            model_state_dict = Model().state_dict()  # Create a state_dict for testing
            export_output.save(tmp_onnx_file.name, model_state_dict=model_state_dict)
            assert (
                len(onnx.load(tmp_onnx_file.name).graph.initializer) == 2
            ), "Initializers must be present after loading it from model_state_dict"

        # Variant 2: Save ONNX proto using Model checkpoint file
        with tempfile.NamedTemporaryFile(
            suffix=".onnx"
        ) as tmp_onnx_file, tempfile.NamedTemporaryFile(
            suffix=".pt"
        ) as tmp_checkpoint_file:
            torch.save(
                Model().state_dict(), tmp_checkpoint_file.name
            )  # Create checkpoint file for testing
            export_output.save(
                tmp_onnx_file.name, model_state_dict=tmp_checkpoint_file.name
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

    def test_fake_tensor_mode_huggingface_gpt2(self):
        config = transformers.GPT2Config()
        batch, seq = 4, 256

        with torch.onnx.enable_fake_mode() as fake_context:
            model = transformers.GPT2Model(config).eval()
            input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            attention_mask = torch.ones(batch, seq, dtype=torch.bool)
            position_ids = torch.arange(0, seq, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).view(-1, seq)

            export_options = torch.onnx.ExportOptions(fake_context=fake_context)
            export_output = torch.onnx.dynamo_export(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                export_options=export_options,
            )
            onnx.checker.check_model(export_output.model_proto)
            onnx.shape_inference.infer_shapes(export_output.model_proto)

    def test_fake_tensor_mode_huggingface_bigscience_bloom(self):
        config = transformers.BloomConfig()
        batch, seq = 4, 256

        with torch.onnx.enable_fake_mode() as fake_context:
            model = transformers.BloomModel(config).eval()
            input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            attention_mask = torch.ones(batch, seq, dtype=torch.bool)

            export_options = torch.onnx.ExportOptions(fake_context=fake_context)
            export_output = torch.onnx.dynamo_export(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                export_options=export_options,
            )
            onnx.checker.check_model(export_output.model_proto)
            onnx.shape_inference.infer_shapes(export_output.model_proto)

    def test_fake_tensor_mode_huggingface_open_llama(self):
        config = transformers.OpenLlamaConfig()
        batch, seq = 4, 256

        with torch.onnx.enable_fake_mode() as fake_context:
            model = transformers.OpenLlamaModel(config).eval()
            input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            attention_mask = torch.ones(batch, seq, dtype=torch.bool)
            position_ids = torch.arange(0, seq, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).view(-1, seq)

            export_options = torch.onnx.ExportOptions(fake_context=fake_context)
            export_output = torch.onnx.dynamo_export(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                export_options=export_options,
            )
            onnx.checker.check_model(export_output.model_proto)
            onnx.shape_inference.infer_shapes(export_output.model_proto)

    # TODO: From Config/Model
    def test_fake_tensor_mode_huggingface_google_t5(self):
        config = transformers.T5Config()
        device = "cpu"
        batch, seq = 4, 256
        with torch.onnx.enable_fake_mode() as fake_context:
            model = transformers.T5Model(config).to(device).eval()
            input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            attention_mask = torch.ones((batch, seq), dtype=torch.bool)
            decoder_input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            export_options = torch.onnx.ExportOptions(fake_context=fake_context)
            export_output = torch.onnx.dynamo_export(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                export_options=export_options,
            )
            onnx.checker.check_model(export_output.model_proto)
            onnx.shape_inference.infer_shapes(export_output.model_proto)

    def test_fake_tensor_mode_huggingface_openai_whisper(self):
        config = transformers.WhisperConfig()
        feature_extractor = transformers.WhisperFeatureExtractor()
        device = "cpu"
        batch = 4
        with torch.onnx.enable_fake_mode() as fake_context:
            input_features = torch.randn(
                (
                    batch,
                    feature_extractor.feature_size,
                    feature_extractor.nb_max_frames,
                ),
                dtype=torch.float32,
            )
            decoder_input_ids = torch.tensor([[1, 1]]) * config.decoder_start_token_id
            model = transformers.AutoModel.from_config(config).to(device).eval()
            export_options = torch.onnx.ExportOptions(fake_context=fake_context)
            export_output = torch.onnx.dynamo_export(
                model,
                input_features,
                decoder_input_ids=decoder_input_ids,
                export_options=export_options,
            )
            onnx.checker.check_model(export_output.model_proto)
            onnx.shape_inference.infer_shapes(export_output.model_proto)

    # TODO: From Config/Model
    @pytorch_test_common.skip_in_ci(
        "Skip this test in CI because of memory issue."
        "SymFloat in OnnxFUnction attribute is not supported yet."
    )
    def test_fake_tensor_mode_huggingface_databricks_dolly_v2_3b(self):
        model_name = "databricks/dolly-v2-3b"
        device = "cpu"
        with torch.onnx.enable_fake_mode() as fake_context:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            inputs = tokenizer("Hello world!", return_tensors="pt")
            model = transformers.AutoModel.from_pretrained(model_name).to(device).eval()

            export_options = torch.onnx.ExportOptions(fake_context=fake_context)
            export_output = torch.onnx.dynamo_export(
                model, **inputs, export_options=export_options
            )
            onnx.checker.check_model(export_output.model_proto)
            onnx.shape_inference.infer_shapes(export_output.model_proto)

    @pytorch_test_common.skip_in_ci(
        "AssertionError: Mutating module attribute seq_len_cached during export."
        "self.seq_len_cached = seq_len"
        "Skip this test in CI because of memory issue."
    )
    def test_fake_tensor_mode_huggingface_tiiuae_falcon(self):
        config = transformers.FalconConfig()
        batch, seq = 4, 256

        with torch.onnx.enable_fake_mode() as fake_context:
            model = transformers.FalconModel(config).eval()
            input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            attention_mask = torch.ones(batch, seq, dtype=torch.bool)

            export_options = torch.onnx.ExportOptions(fake_context=fake_context)
            export_output = torch.onnx.dynamo_export(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                export_options=export_options,
            )
            onnx.checker.check_model(export_output.model_proto)
            onnx.shape_inference.infer_shapes(export_output.model_proto)

    @pytorch_test_common.skip_in_ci(
        "Skip this test in CI because of memory issue."
        "torch._dynamo.exc.UserError: Dynamic control flow is not supported at the moment. "
        "Please use functorch.experimental.control_flow.cond to explicitly capture the control flow"
    )
    def test_fake_tensor_mode_huggingface_mosaicml_mpt_7b(self):
        model_name = "mosaicml/mpt-7b"
        device = "cpu"
        with torch.onnx.enable_fake_mode() as fake_context:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            inputs = tokenizer("Hello world!", return_tensors="pt")
            model = (
                transformers.AutoModelForCausalLM.from_pretrained(
                    model_name, trust_remote_code=True
                )
                .to(device)
                .eval()
            )

            export_options = torch.onnx.ExportOptions(fake_context=fake_context)
            export_output = torch.onnx.dynamo_export(
                model, **inputs, export_options=export_options
            )
            onnx.checker.check_model(export_output.model_proto)
            onnx.shape_inference.infer_shapes(export_output.model_proto)


if __name__ == "__main__":
    common_utils.run_tests()
