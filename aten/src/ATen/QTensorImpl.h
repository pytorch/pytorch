#pragma once

#include <c10/core/TensorImpl.h>
#include <ATen/Quantizer.h>
#include <c10/util/Exception.h>

namespace at {

struct CAFFE2_API QTensorImpl : public c10::TensorImpl {
public:
  QTensorImpl(Storage&& storage, TensorTypeId type_id, bool is_variable, QuantizerPtr quantizer);

  // FIXME: need to find a better return type, maybe std::shared_ptr<at::Quantizer>
  // Fix after deciding on how to expose this in python frontend
  QuantizerPtr quantizer() {
    return quantizer_;
  }

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach() const override {
    auto impl = c10::make_intrusive<QTensorImpl>(Storage(storage()), type_id(), is_variable(), quantizer_);
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
