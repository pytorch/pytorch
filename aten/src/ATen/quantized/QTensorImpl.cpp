#include <ATen/quantized/QTensorImpl.h>

namespace at {

QTensorImpl::QTensorImpl(
    Storage&& storage,
    DispatchKeySet key_set,
    const caffe2::TypeMeta data_type,
    QuantizerPtr quantizer)
    : TensorImpl(std::move(storage), key_set, data_type),
      quantizer_(quantizer) {}

const char* QTensorImpl::tensorimpl_type_name() const {
  return "QTensorImpl";
}

} // namespace at
