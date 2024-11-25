# Owner(s): ["module: onnx"]
from __future__ import annotations

import logging
import tempfile

import onnx
import onnx.inliner

import pytorch_test_common
import transformers  # type: ignore[import]

import torch
from torch import nn
from torch._subclasses import fake_tensor
from torch.nn import functional as F
from torch.onnx import dynamo_export, ExportOptions
from torch.testing._internal import common_utils


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

    def test_mnist_exported_with_no_warnings(self):
        class MNISTModel(nn.Module):
            def __init__(self) -> None:
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
        onnx_program = dynamo_export(MNISTModel(), tensor_x)
        assert onnx_program is not None

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

        _ = dynamo_export(TopKModel(), x, export_options=self.export_options)

    def test_dynamo_export_retains_readable_parameter_and_buffer_names(self):
        class SubModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
                self.fc1 = nn.Linear(9216, 128, bias=False)
                self.buffer = torch.nn.Buffer(torch.randn(1, 128))

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
            def __init__(self) -> None:
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
            onnx_program = torch.onnx.dynamo_export(
                model, x, export_options=export_options
            )

        assert (
            onnx_program is not None
        ), "ONNXProgram must be created on successful export"

        onnx_program.apply_weights(Model().state_dict())

        assert (
            onnx_program.model_proto is not None
        ), "A model protobuf must be created on a successful export"
        onnx.checker.check_model(onnx_program.model_proto, full_check=True)

    def test_exported_program_torch_distributions_normal_Normal(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
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
            onnx_program.save(
                tmp_onnx_file.name,
                keep_initializers_as_inputs=True,
                include_initializers=False,
            )
            onnx.checker.check_model(tmp_onnx_file.name, full_check=True)

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
