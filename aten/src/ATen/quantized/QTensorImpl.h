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
    copy_tensor_metadata(this, impl.get());
    impl->refresh_numel();
    impl->refresh_contiguous();

    // QTensorImpl-specific fields (none currently).
    return impl;
  }

  /**
   * Shallow-copies data from another TensorImpl into this TensorImpl.
   * See NOTE [ TensorImpl Shallow-Copying ] for details.
   */
  void shallow_copy_from(c10::intrusive_ptr<TensorImpl> impl) override {
    copy_tensor_metadata(impl.get(), this);

    // QTensorImpl-specific fields
    auto q_impl = static_cast<QTensorImpl*>(impl.get());
    set_storage(q_impl->storage());
    quantizer_ = q_impl->quantizer();

    refresh_numel();
    refresh_contiguous();
  }

 private:
  QuantizerPtr quantizer_;
};

} // namespace at
