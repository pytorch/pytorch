#include <ATen/quantized/QTensorImpl.h>

namespace at {

QTensorImpl::QTensorImpl(
    Storage&& storage,
    TensorTypeId type_id,
    QuantizerPtr quantizer)
    : TensorImpl(std::move(storage), type_id),
      quantizer_(quantizer) {}

} // namespace at
