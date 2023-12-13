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

template <>
struct ThreadLocalCachedOutputTensor<ConstantHandle> {
  explicit ThreadLocalCachedOutputTensor(const ConstantHandle&) {}
  void copy_data_from(const ConstantHandle& handle) {
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

template <typename T>
struct ThreadLocalCachedOutputArray;

// Just needs to compile, doesn't need to do anything.
template <>
struct ThreadLocalCachedOutputArray<RAIIAtenTensorHandle> {
  explicit ThreadLocalCachedOutputArray(const RAIIAtenTensorHandle&) {
    throw std::runtime_error("can't happen");
  }

  // Not supported yet! We would need to put contiguous() or
  // expect_contiguous() into the ABI.
  void copy_data_from(const RAIIAtenTensorHandle&) {
    throw std::runtime_error("can't happen");
  }

  template <typename U>
  ArrayRefTensor<U> arrayref_tensor() const {
    throw std::runtime_error("can't happen");
  }
};

// Just needs to compile, doesn't need to do anything.
template <>
struct ThreadLocalCachedOutputArray<ConstantHandle> {
  explicit ThreadLocalCachedOutputArray(const ConstantHandle&) {
    throw std::runtime_error("can't happen");
  }

  // Not supported yet! We would need to put contiguous() or
  // expect_contiguous() into the ABI.
  void copy_data_from(const ConstantHandle&) {
    throw std::runtime_error("can't happen");
  }

  template <typename U>
  ArrayRefTensor<U> arrayref_tensor() const {
    throw std::runtime_error("can't happen");
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
      storage_ = std::make_unique<T[]>(capacity_);
    }
    std::copy(t.data(), t.data() + t.numel(), storage_.get());
    tensor_ = t;
    tensor_.set_arrayref(MiniArrayRef<T>(storage_.get(), t.numel()));
  }

 private:
  std::unique_ptr<T[]> storage_;
  uint32_t capacity_ = 0;
  ArrayRefTensor<T> tensor_;
};

template <typename T>
void convert_output_to_handle(
    const ArrayRefTensor<T>& output,
    AtenTensorHandle& handle) {
  handle = output.expensiveCopyToTensor();
}

template <typename... Ts, std::size_t... Is>
void convert_outputs_to_handles_helper(
    const std::tuple<ArrayRefTensor<Ts>...>& outputs,
    AtenTensorHandle* output_handles,
    std::index_sequence<Is...>) {
  (convert_output_to_handle(std::get<Is>(outputs), output_handles[Is]), ...);
}
template <typename... Ts>
void convert_outputs_to_handles(
    const std::tuple<ArrayRefTensor<Ts>...>& outputs,
    AtenTensorHandle* output_handles) {
  convert_outputs_to_handles_helper(
      outputs, output_handles, std::make_index_sequence<sizeof...(Ts)>());
}

template <typename T>
void convert_handle_to_arrayref_tensor(
    AtenTensorHandle handle,
    ArrayRefTensor<T>& input) {
  void* data_ptr;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(handle, &data_ptr));
  int64_t dim;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dim(handle, &dim));
  int64_t numel;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_numel(handle, &numel));
  int64_t* sizes;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(handle, &sizes));
  int64_t* strides;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(handle, &strides));
  int32_t dtype;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(handle, &dtype));
  int32_t device_type;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(handle, &device_type));
  int32_t device_index;
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_index(handle, &device_index));

  input = ArrayRefTensor<T>(
      MiniArrayRef<T>(reinterpret_cast<T*>(data_ptr), numel),
      MiniArrayRef<const int64_t>(sizes, dim),
      MiniArrayRef<const int64_t>(strides, dim),
      dtype,
      device_type,
      device_index);
}

template <typename... Ts, std::size_t... Is>
void convert_handles_to_inputs_helper(
    AtenTensorHandle* input_handles,
    std::tuple<ArrayRefTensor<Ts>...>& inputs,
    std::index_sequence<Is...>) {
  (convert_handle_to_arrayref_tensor(input_handles[Is], std::get<Is>(inputs)),
   ...);
}

template <typename... Ts>
void convert_handles_to_inputs(
    AtenTensorHandle* input_handles,
    std::tuple<ArrayRefTensor<Ts>...>& inputs) {
  convert_handles_to_inputs_helper(
      input_handles, inputs, std::make_index_sequence<sizeof...(Ts)>());
}

template <typename T>
const T& convert_arrayref_tensor_to_tensor(const T& t) {
  return t;
}

template <typename T>
RAIIAtenTensorHandle convert_arrayref_tensor_to_tensor(
    const ArrayRefTensor<T>& art) {
  return art.expensiveCopyToTensor();
}

} // namespace aot_inductor
} // namespace torch
