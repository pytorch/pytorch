#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/Functions.h>
#include <ATen/RedispatchFunctions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/FunctionalInverses.h>
#include <torch/library.h>

// ${generated_comment}

namespace at {
namespace functionalization {


${func_definitions}
}  // namespace func

namespace {

TORCH_LIBRARY_IMPL(aten, Functionalize, m) {
  ${func_registrations};
}

}  // namespace

} // namespace at
