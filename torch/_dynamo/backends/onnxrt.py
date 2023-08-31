# This backend is maintained by ONNX team. To direct issues
# to the right people, please tag related GitHub issues with `module: onnx`.
#
# Maintainers' Github IDs: wschin, thiagocrepaldi, BowenBao, abock
from torch.onnx._internal.onnxruntime import has_onnxruntime, make_aot_ort
from .registry import register_backend

if has_onnxruntime():
    aot_ort, ort = make_aot_ort(dynamic=True)
    register_backend(name="onnxrt", compiler_fn=aot_ort)
else:

    def information_displaying_backend(*args, **kwargs):
        raise ImportError(
            "onnxrt is not registered as a backend. "
            "Please make sure all dependencies such as "
            "numpy, onnx, onnxscript, and onnxruntime-training are installed. "
            "Suggested procedure to fix dependency problem: "
            "(1) pip or conda install numpy onnx onnxscript onnxruntime-training. "
            "(2) open a new python terminal "
            "(3) Run `from torch.onnx._internal.onnxruntime import has_onnxruntime`. "
            "(4) Run `has_onnxruntime()`. "
            "(5) If has_onnxruntime() returns True, then you can use `onnxrt` backend. "
            "(6) If has_onnxruntime() returns False, please execute the package importing section in "
            "torch/onnx/_internal/onnxruntime.py under pdb line-by-line to see which import fails."
        )

    register_backend(name="onnxrt", compiler_fn=information_displaying_backend)
