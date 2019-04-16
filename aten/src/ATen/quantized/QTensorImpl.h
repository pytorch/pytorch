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

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   * See NOTE [ TensorImpl Shallow-Copying ] for details.
   */
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

  /**
   * Shallow-copies data from another TensorImpl into this TensorImpl.
   * See NOTE [ TensorImpl Shallow-Copying ] for details.
   */
  void shallow_copy_from(c10::intrusive_ptr<TensorImpl> impl) override {
    auto q_impl = static_cast<QTensorImpl*>(impl.get());
    set_storage(q_impl->storage());
    type_id_ = q_impl->type_id();
    quantizer_ = q_impl->quantizer();
    set_sizes_and_strides(q_impl->sizes(), q_impl->strides());
    set_storage_offset(q_impl->storage_offset());
    is_wrapped_number_ = q_impl->is_wrapped_number_;
    reserved_ = q_impl->reserved_;
    refresh_numel();
    refresh_contiguous();
  }

 private:
  QuantizerPtr quantizer_;
};

} // namespace at
