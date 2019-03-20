#include <ATen/native/TensorFactories.h>
#include <ATen/Quantizer.h>
#include <ATen/QTensorImpl.h>

namespace at {
namespace native {

// TODO: add quantizer as an argument
Tensor empty_quantized(IntArrayRef sizes, const TensorOptions& options, float scale, int32_t zero_point) {
  AT_ASSERT(options.backend() == Backend::Quantized);
  // Must specify QScheme
  // AT_ASSERT(options.has_qscheme());
  // TODO: change this to a function call to check whether the dtype is allowed
  // for quantized backend?
  AT_ASSERT(typeMetaToScalarType(options.dtype()) == ScalarType::QInt8);
  AT_ASSERT(!options.is_variable());  // is_variable should have been 'unpacked'  // TODO: remove this when Variable and Tensor are merged
  return new_qtensor(sizes, options, scale, zero_point);
}

} // namespace native
} // namespace at
