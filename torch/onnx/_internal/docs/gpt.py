# Import generic wrappers
import transformers
from transformers import AutoModel, AutoTokenizer
import itertools
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.nn.utils import stateless
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
import torch
from torch import _dynamo as torchdynamo
from torch.fx.experimental import proxy_tensor
import onnx

# Define the model repo
model_name = "sshleifer/tiny-gpt2"

# Download pytorch model
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Transform input tokens
inputs = tokenizer("Hello world!", return_tensors="pt")

# Model apply
outputs = model(**inputs)

# Export tiny GPT2.
# from torch.onnx._internal._fx import export
from torch.onnx._internal import _fx as fx_onnx
#onnx_model = export(model, **inputs)
input_ids = inputs["input_ids"]
# onnx_model = export(model, input_ids)
#attention_mask = inputs["attention_mask"]
#onnx_model = export(model, input_ids, attention_mask=attention_mask)


def test_gpt2_one_shot(model, input_ids):
    fx_model = proxy_tensor.make_fx(model, fx_onnx._ONNX_FRIENDLY_DECOMPOSITION_TABLE)(input_ids)
    fx_model.print_readable()

    # Use this mode to
    # 1. convert nn.Parameter's in nn.Module to FakeTensor
    # 2. run FakeTensorProp
    fake_tensor_mode = FakeTensorMode()

    def to_fake_tensor(x):
        if isinstance(x, torch.Tensor) and not isinstance(x, FakeTensor):
            return fake_tensor_mode.from_tensor(x)
        return x

    # "args" are FakeTensor in FakeTensorProp so the parameters and buffers
    # in model must be converted to FakeTensor as well.
    fake_parameters_and_buffers = {
        k: to_fake_tensor(v)
        for k, v in itertools.chain(fx_model.named_parameters(), fx_model.named_buffers())
    }

    # Shape inference via FakeTensorProp
    with stateless._reparametrize_module(
        fx_model, fake_parameters_and_buffers
    ):
        # Assign output types and shapes to each node.
        # TODO(wechi): It's possible to get symbolic types (and shapes)
        # for each node's output. Consider to set "tracing_mode=symbolic"
        # when calling make_fx and then remove FakeTensorProp below.
        FakeTensorProp(fx_model, fake_tensor_mode).propagate(input_ids)

    ts_graph, ts_initializers = fx_onnx._export_fx_to_ts(fx_model)
    onnx_model = fx_onnx._ts_graph_to_onnx_model_in_protobuf(ts_graph, ts_initializers)
    model_proto = onnx.ModelProto.FromString(onnx_model)
    print(model_proto)


def test_gpt2_auto_regressive(model_name, input_ids):
    # NOTE: auto regressive uses generation algorithms such as greedy search or beam
    # search that involves loops and control flows.

    model = transformers.GPT2LMHeadModel.from_pretrained(model_name)

    (
        explanation,
        out_guards,
        graphs,
        ops_per_graph,
        break_reasons,
        explanation_verbose
    ) = torchdynamo.explain(model.generate, input_ids)

    print(explanation_verbose)


test_gpt2_auto_regressive(model_name, input_ids)
