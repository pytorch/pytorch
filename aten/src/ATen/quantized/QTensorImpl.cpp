#include <ATen/quantized/QTensorImpl.h>

namespace at {

QTensorImpl::QTensorImpl(
    Storage&& storage,
    DispatchKeySet key_set,
    QuantizerPtr quantizer)
    : TensorImpl(std::move(storage), key_set),
      quantizer_(quantizer) {}

} // namespace at
