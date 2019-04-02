#include <ATen/quantized/QTensorImpl.h>

namespace at {

QTensorImpl::QTensorImpl(
    Storage&& storage,
    TensorTypeId type_id,
    bool is_variable,
    QuantizerPtr quantizer)
    : TensorImpl(std::move(storage), type_id, is_variable),
      quantizer_(quantizer) {}

} // namespace at
