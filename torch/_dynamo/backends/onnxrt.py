from torch.onnx._backend.core import has_onnxruntime, make_aot_ort
from .registry import register_backend

if has_onnxruntime():
    aot_ort, ort = make_aot_ort(dynamic=True)
    register_backend(name="onnxrt", compiler_fn=aot_ort)
