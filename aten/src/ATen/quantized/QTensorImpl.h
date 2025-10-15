#pragma once

#include <ATen/quantized/Quantizer.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>

namespace at {

/**
 * QTensorImpl is a TensorImpl for Quantized Tensors, it stores Quantizer which
 * specifies the quantization scheme and parameters, for more information please
 * see ATen/quantized/Quantizer.h
 *
 * We'll use QTensor in code or documentation to refer to a Tensor with QTensorImpl.
 */
struct TORCH_API QTensorImpl : public c10::TensorImpl {
 public:
  QTensorImpl(
      Storage&& storage,
      DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      QuantizerPtr quantizer);

  // See Note [Enum ImplType]
  QTensorImpl(
      ImplType type,
      Storage&& storage,
      DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      QuantizerPtr quantizer);


  // TODO: Expose in PyTorch Frontend
  QuantizerPtr quantizer() {
    return quantizer_;
  }

  void set_quantizer_(QuantizerPtr quantizer) {
    quantizer_ = quantizer;
  }

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<QTensorImpl>(
        Storage(storage()), key_set(), data_type_, quantizer_);
    copy_tensor_metadata(
      /*src_q_impl=*/this,
      /*dest_q_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    impl->refresh_numel();
    impl->refresh_contiguous();
    return impl;
  }

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<QTensorImpl>(
        Storage(storage()), key_set(), data_type_, quantizer_);
    copy_tensor_metadata(
      /*src_q_impl=*/this,
      /*dest_q_impl=*/impl.get(),
      /*version_counter=*/std::move(version_counter),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    impl->refresh_numel();
    impl->refresh_contiguous();
    return impl;
  }

  /**
   * Shallow-copies data from another TensorImpl into this TensorImpl.
   *
   * For why this function doesn't check this TensorImpl's `allow_tensor_metadata_change_`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override {
    AT_ASSERT(has_compatible_shallow_copy_type(impl->key_set()));
    auto q_impl = static_cast<const QTensorImpl*>(impl.get());
    copy_tensor_metadata(
      /*src_q_impl=*/q_impl,
      /*dest_q_impl=*/this,
      /*version_counter=*/version_counter(),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
    refresh_numel();
    refresh_contiguous();
  }

 private:
  QuantizerPtr quantizer_;

  const char* tensorimpl_type_name() const override;

  /**
   * Copy the tensor metadata fields (e.g. sizes / strides / storage pointer / storage_offset)
   * from one TensorImpl to another TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`, see NOTE [ TensorImpl Shallow-Copying ].
   */
  static void copy_tensor_metadata(
      const QTensorImpl* src_q_impl,
      QTensorImpl* dest_q_impl,
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) {
    TensorImpl::copy_tensor_metadata(src_q_impl, dest_q_impl, version_counter, allow_tensor_metadata_change);

    // OpaqueTensorImpl-specific fields.
    dest_q_impl->quantizer_ = src_q_impl->quantizer_;
  }
};

} // namespace at
