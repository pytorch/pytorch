#include <ATen/quantized/QTensorImpl.h>

namespace at {

QTensorImpl::QTensorImpl(
    Storage&& storage,
    DispatchKeySet key_set,
    const caffe2::TypeMeta data_type,
    QuantizerPtr quantizer)
    : TensorImpl(std::move(storage), std::move(key_set), data_type),
      quantizer_(std::move(quantizer)) {}

QTensorImpl::QTensorImpl(
    ImplType type,
    Storage&& storage,
    DispatchKeySet key_set,
    const caffe2::TypeMeta data_type,
    QuantizerPtr quantizer)
    : TensorImpl(type, std::move(storage), std::move(key_set), data_type),
      quantizer_(std::move(quantizer)) {}

const char* QTensorImpl::tensorimpl_type_name() const {
  return "QTensorImpl";
}

} // namespace at
