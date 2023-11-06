// NOTE: Like interface.cpp, this file will be copied into AOTInductor
// generated output. This file is intended to keep implementation
// details separate from the implementation of the AOTI public
// interface. Note also that #includes should go into interface.cpp
// for simplicity of maintenance.

namespace torch {
namespace aot_inductor {
template <typename T>
struct ThreadLocalCachedOutputTensor;

template <>
struct ThreadLocalCachedOutputTensor<RAIIAtenTensorHandle> {
  explicit ThreadLocalCachedOutputTensor(const RAIIAtenTensorHandle&) {}
  void copy_data_from(const RAIIAtenTensorHandle& handle) {
    throw std::runtime_error("can't happen");
  }

  AtenTensorHandle tensor() const {
    throw std::runtime_error("can't happen");
  }
};

template <>
struct ThreadLocalCachedOutputTensor<AtenTensorHandle> {
  explicit ThreadLocalCachedOutputTensor(const AtenTensorHandle&) {}
  void copy_data_from(const AtenTensorHandle& handle) {
    throw std::runtime_error("can't happen");
  }

  AtenTensorHandle tensor() const {
    throw std::runtime_error("can't happen");
  }
};

template <typename T>
struct ThreadLocalCachedOutputTensor<ArrayRefTensor<T>> {
  explicit ThreadLocalCachedOutputTensor(const ArrayRefTensor<T>& t) {
    realloc(t);
  }

  void copy_data_from(const ArrayRefTensor<T>& t) {
    if (t.numel() > capacity_) {
      realloc(t);
    }
    std::copy(t.data(), t.data() + t.numel(), storage_.get());
  }

  AtenTensorHandle tensor() const {
    return tensor_.get();
  }

 private:
  void realloc(const ArrayRefTensor<T>& t) {
    capacity_ = t.numel();
    storage_ = std::make_unique<T[]>(t.numel());
    AtenTensorHandle handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_create_tensor_from_blob(
        storage_.get(),
        t.sizes().size(),
        t.sizes().data(),
        t.strides().data(),
        0,
        t.dtype(),
        t.device_type(),
        t.device_idx(),
        &handle));
    tensor_ = handle;
  }

  std::unique_ptr<T[]> storage_;
  size_t capacity_ = 0;
  RAIIAtenTensorHandle tensor_;
};

} // namespace aot_inductor
} // namespace torch
