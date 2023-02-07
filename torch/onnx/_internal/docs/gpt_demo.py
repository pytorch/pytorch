import transformers
import torch
from torch.onnx._internal import fx as fx_onnx, diagnostics
from torch.utils import _pytree as pytree
import onnx
import onnx.reference
from typing import Union, Tuple, Any, Sequence
import io


def _run_onnx_reference_runtime(
    onnx_model: Union[str, io.BytesIO],
    pytorch_inputs: Tuple[Any, ...],
    verbose: int = 10,
) -> Sequence[Any]:
    session = onnx.reference.ReferenceEvaluator(onnx_model, verbose=verbose)
    return session.run(
        None, {k: v.cpu().numpy() for k, v in zip(session.input_names, pytorch_inputs)}
    )


model_name = "sshleifer/tiny-gpt2"
# Download pytorch model
model = transformers.AutoModel.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# Transform input tokens
inputs = tokenizer("Hello world!", return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

with diagnostics.engine.create_diagnostic_context("fx-exporter", version=torch.__version__):
    onnx_model = fx_onnx.export_without_kwargs(model, 17, **inputs, use_binary_format=True)
diagnostics.engine.dump("report_gpt2_tiny.sarif")

ref_outputs, _ = pytree.tree_flatten(model(**inputs, return_dict=False))
ort_outputs = _run_onnx_reference_runtime(onnx_model, (input_ids, attention_mask))
assert len(ref_outputs) == len(ort_outputs)
assert len(ref_outputs) == 5
for ref_output, ort_output in zip(ref_outputs, ort_outputs):
    torch.testing.assert_close(ref_output, torch.tensor(ort_output))
