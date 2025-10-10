#pragma once

#include <torch/csrc/inductor/aoti_runtime/arrayref_tensor.h>

namespace torch::aot_inductor {

template <typename T>
struct ThreadLocalCachedOutputTensor;

template <>
struct ThreadLocalCachedOutputTensor<RAIIAtenTensorHandle> {
  explicit ThreadLocalCachedOutputTensor(const RAIIAtenTensorHandle&) {}
  void copy_data_from(const RAIIAtenTensorHandle& handle) {
    STD_TORCH_CHECK(false, "can't happen");
  }

  AtenTensorHandle tensor() const {
    STD_TORCH_CHECK(false, "can't happen");
  }
};

template <>
struct ThreadLocalCachedOutputTensor<AtenTensorHandle> {
  explicit ThreadLocalCachedOutputTensor(const AtenTensorHandle&) {}
  void copy_data_from(const AtenTensorHandle& handle) {
    STD_TORCH_CHECK(false, "can't happen");
  }

  AtenTensorHandle tensor() const {
    STD_TORCH_CHECK(false, "can't happen");
  }
};

template <>
struct ThreadLocalCachedOutputTensor<ConstantHandle> {
  explicit ThreadLocalCachedOutputTensor(const ConstantHandle&) {}
  void copy_data_from(const ConstantHandle& handle) {
    STD_TORCH_CHECK(false, "can't happen");
  }

  AtenTensorHandle tensor() const {
    STD_TORCH_CHECK(false, "can't happen");
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
    // NOLINTNEXTLINE(*arrays*)
    storage_ = std::make_unique<T[]>(t.numel());
    AtenTensorHandle handle = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_create_tensor_from_blob(
        storage_.get(),
        t.sizes().size(),
        t.sizes().data(),
        t.strides().data(),
        0,
        aoti_torch_dtype<std::remove_const_t<T>>(),
        t.device_type(),
        t.device_idx(),
        &handle));
    tensor_ = handle;
  }

  // NOLINTNEXTLINE(*arrays*)
  std::unique_ptr<T[]> storage_;
  int64_t capacity_ = 0;
  RAIIAtenTensorHandle tensor_;
};

template <typename T>
struct ThreadLocalCachedOutputArray;

// Just needs to compile, doesn't need to do anything.
template <>
struct ThreadLocalCachedOutputArray<RAIIAtenTensorHandle> {
  explicit ThreadLocalCachedOutputArray(const RAIIAtenTensorHandle&) {
    STD_TORCH_CHECK(false, "can't happen");
  }

  // Not supported yet! We would need to put contiguous() or
  // expect_contiguous() into the ABI.
  void copy_data_from(const RAIIAtenTensorHandle&) {
    STD_TORCH_CHECK(false, "can't happen");
  }

  template <typename U>
  ArrayRefTensor<U> arrayref_tensor() const {
    STD_TORCH_CHECK(false, "can't happen");
  }
};

// Just needs to compile, doesn't need to do anything.
template <>
struct ThreadLocalCachedOutputArray<ConstantHandle> {
  explicit ThreadLocalCachedOutputArray(const ConstantHandle&) {
    STD_TORCH_CHECK(false, "can't happen");
  }

  // Not supported yet! We would need to put contiguous() or
  // expect_contiguous() into the ABI.
  void copy_data_from(const ConstantHandle&) {
    STD_TORCH_CHECK(false, "can't happen");
  }

  template <typename U>
  ArrayRefTensor<U> arrayref_tensor() const {
    STD_TORCH_CHECK(false, "can't happen");
  }
};

template <typename T>
struct ThreadLocalCachedOutputArray<ArrayRefTensor<T>> {
  explicit ThreadLocalCachedOutputArray(const ArrayRefTensor<T>& t) {}

  template <
      typename U,
      std::enable_if_t<
          std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>>,
          bool> = true>
  ArrayRefTensor<T> arrayref_tensor() const {
    return tensor_;
  }

  void copy_data_from(const ArrayRefTensor<T>& t) {
    if (t.numel() > capacity_) {
      capacity_ = t.numel();
      // NOLINTNEXTLINE(*arrays*)
      storage_ = std::make_unique<T[]>(capacity_);
    }
    std::copy(t.data(), t.data() + t.numel(), storage_.get());
    tensor_ = t;
    tensor_.set_arrayref(MiniArrayRef<T>(storage_.get(), t.numel()));
  }

 private:
  // NOLINTNEXTLINE(*arrays*)
  std::unique_ptr<T[]> storage_;
  uint32_t capacity_ = 0;
  ArrayRefTensor<T> tensor_;
};

} // namespace torch::aot_inductor
