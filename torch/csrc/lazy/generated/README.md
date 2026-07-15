This folder contains generated sources for the lazy torchscript backend.

The main input file that drives which operators get codegen support for torchscript backend is
[../../../../aten/src/ATen/native/ts_native_functions.yaml](../../../../aten/src/ATen/native/ts_native_functions.yaml)

The code generator lives at `torchgen/gen_lazy_tensor.py`.

It is called automatically by the torch autograd codegen (`tools/setup_helpers/generate_code.py`)
as a part of the build process in OSS builds (CMake/Bazel) and Buck.

External backends (e.g. torch/xla) call `gen_lazy_tensor.py` directly,
and feed it command line args indicating where the output files should go.

For more information on codegen, see these resources:
* Info about lazy tensor codegen: [gen_lazy_tensor.py docs](../../../../torchgen/gen_lazy_tensor.py)
* Lazy TorchScript backend native functions: [ts_native_functions.yaml](../../../../aten/src/ATen/native/ts_native_functions.yaml)
* Source of truth for native func definitions [ATen native_functions.yaml](../../../../aten/src/ATen/native/native_functions.yaml)
* Info about native functions [ATen nativefunc README.md](../../../../aten/src/ATen/native/README.md)
