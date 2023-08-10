# This backend is maintained by ONNX team. To direct issues
# to the right people, please tag related GitHub issues with `onnx`.
#
# Maintainers' Github IDs: wschin, thiagocrepaldi, BowenBao, abock
from torch.onnx._internal.onnxruntime import has_onnxruntime, make_aot_ort
from .registry import register_backend

if has_onnxruntime():
    aot_ort, ort = make_aot_ort(dynamic=True)
    register_backend(name="onnxrt", compiler_fn=aot_ort)
