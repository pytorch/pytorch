#include <ATen/quantized/QTensorImpl.h>

namespace at {

QTensorImpl::QTensorImpl(
    Storage&& storage,
    TensorTypeSet type_set,
    QuantizerPtr quantizer)
    : TensorImpl(std::move(storage), type_set),
      quantizer_(quantizer) {}

} // namespace at
