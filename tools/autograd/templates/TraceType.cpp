#include "torch/csrc/autograd/VariableTypeUtils.h"

#include <torch/library.h>

#include "torch/csrc/autograd/function.h"

#include <ATen/RedispatchFunctions.h>
#include "ATen/quantized/Quantizer.h"

// ${generated_comment}

// See the `Tracer` section in `torch/csrc/jit/OVERVIEW.md`.
// NOTE See [Sharded File] comment in VariableType

using namespace at;

namespace torch {

namespace TraceType {

namespace {
${trace_method_definitions}
}  // namespace
}  // namespace TraceType

namespace {

TORCH_LIBRARY_IMPL(aten, Tracer, m) {
  ${trace_wrapper_registrations};
}

}  // namespace

} // namespace torch
