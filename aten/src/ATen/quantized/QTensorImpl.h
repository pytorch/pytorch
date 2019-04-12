#pragma once

#include <ATen/quantized/Quantizer.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>

namespace at {

struct CAFFE2_API QTensorImpl : public c10::TensorImpl {
 public:
  QTensorImpl(
      Storage&& storage,
      TensorTypeId type_id,
      QuantizerPtr quantizer);

  // TODO: Expose in PyTorch Frontend
  QuantizerPtr quantizer() {
    return quantizer_;
  }

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach() const override {
    auto impl = c10::make_intrusive<QTensorImpl>(
        Storage(storage()), type_id(), quantizer_);
    impl->set_sizes_and_strides(sizes(), strides());
    impl->storage_offset_ = storage_offset_;
    impl->is_wrapped_number_ = is_wrapped_number_;
    impl->reserved_ = reserved_;
    impl->refresh_numel();
    impl->refresh_contiguous();
    return impl;
  }

 private:
  QuantizerPtr quantizer_;
};

} // namespace at
