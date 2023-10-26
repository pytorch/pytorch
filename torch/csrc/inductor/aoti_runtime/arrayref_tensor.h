#pragma once

#include <c10/util/ArrayRef.h>
#include <torch/csrc/inductor/aoti_runtime/model.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <assert.h>
#include <cstdint>

namespace torch {
namespace aot_inductor {

// Shim for AOTI generated code to pretend a raw array works like an
// AtenTensorHandle.
template <typename T, size_t N>
class ArrayRefTensor {
 public:
  explicit ArrayRefTensor(
      c10::ArrayRef<T> arr,
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides,
      int32_t dtype,
      int32_t device_type,
      int32_t device_idx)
      : arrayRef_(arr),
        sizes_(sizes),
        strides_(strides),
        dtype_(dtype),
        device_type_(device_type),
        device_idx_(device_idx) {
    assert(arr.size() == N);
  }

  AtenTensorHandle expensiveCopyToTensor() const {
    AtenTensorHandle result;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
        sizes_.size(),
        sizes_.data(),
        strides_.data(),
        dtype_,
        device_type_,
        device_idx_,
        &result));
    void* dataPtr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(result, &dataPtr));
    memcpy(dataPtr, data(), numel() * sizeof(T));
    return result;
  }

  // We need to look the same as RAIIAtenTensorHandle, which returns
  // an owning AtenTensorHandle from release(). So, we allocate one!
  AtenTensorHandle release() {
    return expensiveCopyToTensor();
  }

  // We don't need to free any memory.
  void reset() {}

  auto sizes() const {
    return sizes_;
  }

  auto strides() const {
    return strides_;
  }

  auto dtype() const {
    return dtype_;
  }

  auto device_type() const {
    return device_type_;
  }

  auto device_idx() const {
    return device_idx_;
  }

  T* data() {
    return const_cast<T*>(arrayRef_.data());
  }

  const T* data() const {
    return arrayRef_.data();
  }

  static constexpr auto numel() {
    return N;
  }

 private:
  c10::ArrayRef<T> arrayRef_;
  // We expect generated code to have statically available sizes &
  // strides for us.
  c10::IntArrayRef sizes_;
  c10::IntArrayRef strides_;
  int32_t dtype_;
  int32_t device_type_;
  int32_t device_idx_;
};

inline AtenTensorHandle reinterpret_tensor_wrapper(
    AtenTensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset) {
  AtenTensorHandle result;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__reinterpret_tensor(
      self, ndim, sizes_ptr, strides_ptr, storage_offset, &result));
  return result;
}

inline bool is_contiguous_strides_for_shape(
    int64_t ndim,
    const int64_t* strides_ptr,
    const int64_t* sizes_ptr) {
  int64_t z = 1;
  for (int64_t d = ndim - 1; d >= 0; d--) {
    const auto& size_d = sizes_ptr[d];
    if (size_d != 1) {
      if (strides_ptr[d] == z) {
        z *= size_d;
      } else {
        return false;
      }
    }
  }
  return true;
}

template <typename T, size_t N>
inline ArrayRefTensor<T, N> reinterpret_tensor_wrapper(
    const ArrayRefTensor<T, N>& self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset) {
  // REVIEW: we should add a way to build the DSO in debug mode during
  // tests so we can have checks like this!
  assert(is_contiguous_strides_for_shape(ndim, strides_ptr, sizes_ptr));
  return ArrayRefTensor<T, N>(
      c10::ArrayRef<T>(
          self.data() + storage_offset, self.numel() - storage_offset),
      c10::IntArrayRef(sizes_ptr, ndim),
      c10::IntArrayRef(strides_ptr, ndim),
      self.dtype(),
      self.device_type(),
      self.device_idx());
}

inline void* get_data_ptr_wrapper(AtenTensorHandle tensor) {
  void* result;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(tensor, &result));
  return result;
}

template <typename T, size_t N>
inline T* get_data_ptr_wrapper(ArrayRefTensor<T, N>& tensor) {
  return tensor.data();
}

inline AtenTensorHandle unwrap_raii_handle_if_needed(
    const RAIIAtenTensorHandle& handle) {
  return handle.get();
}

template <typename T, size_t N>
inline const ArrayRefTensor<T, N>& unwrap_raii_handle_if_needed(
    const ArrayRefTensor<T, N>& tensor) {
  return tensor;
}

template <typename T, size_t N>
inline ArrayRefTensor<T, N>& unwrap_raii_handle_if_needed(
    ArrayRefTensor<T, N>& tensor) {
  return tensor;
}

inline RAIIAtenTensorHandle wrap_with_raii_handle_if_needed(
    AtenTensorHandle handle) {
  return RAIIAtenTensorHandle(handle);
}

template <typename T, size_t N>
inline const ArrayRefTensor<T, N>& wrap_with_raii_handle_if_needed(
    const ArrayRefTensor<T, N>& tensor) {
  return tensor;
}

template <typename T, size_t N>
inline ArrayRefTensor<T, N>& wrap_with_raii_handle_if_needed(
    ArrayRefTensor<T, N>& tensor) {
  return tensor;
}

template <typename T, size_t N>
inline RAIIAtenTensorHandle expensive_copy_to_tensor_if_needed(
    const ArrayRefTensor<T, N>& tensor) {
  return tensor.expensiveCopyToTensor();
}

inline AtenTensorHandle expensive_copy_to_tensor_if_needed(
    AtenTensorHandle handle) {
  return handle;
}

} // namespace aot_inductor
} // namespace torch
