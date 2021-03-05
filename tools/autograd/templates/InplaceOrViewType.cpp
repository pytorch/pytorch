#include "torch/csrc/autograd/VariableTypeUtils.h"

#include <torch/library.h>

#include "torch/csrc/autograd/function.h"

#include <ATen/RedispatchFunctions.h>
#include "ATen/quantized/Quantizer.h"

// ${generated_comment}


using namespace at;

namespace torch {

namespace InplaceOrView {

namespace {
${inplace_or_view_method_definitions}
}  // namespace
}  // namespace InplaceOrView

namespace {

TORCH_LIBRARY_IMPL(aten, InplaceOrView, m) {
  ${inplace_or_view_wrapper_registrations};
}

}  // namespace
} // namespace torch

