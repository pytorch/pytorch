#include <ATen/EmptyTensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <ATen/ops/quantize_per_tensor_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <ATen/ops/set_native.h>

#include <c10/core/Allocator.h>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/jit/serialization/pickler.h>

#include <torch/library.h>
