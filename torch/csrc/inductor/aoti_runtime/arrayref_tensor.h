#pragma once

#include <torch/csrc/inductor/aoti_runtime/mini_array_ref.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <cassert>
#include <cstdint>
#include <cstring>

namespace torch::aot_inductor {

using MiniIntArrayRef = MiniArrayRef<int64_t>;

static_assert(
    sizeof(MiniIntArrayRef) == sizeof(void*) + sizeof(size_t),
    "changing the size of MiniArrayRef breaks ABI compatibility!");

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

// Shim for AOTI generated code to pretend a raw array works like an
// AtenTensorHandle.
template <typename T>
class ArrayRefTensor {
 public:
  ArrayRefTensor() = default;

  explicit ArrayRefTensor(
      MiniArrayRef<T> arr,
      MiniArrayRef<const int64_t> sizes,
      MiniArrayRef<const int64_t> strides,
      int32_t device_type,
      int32_t device_idx)
      : arrayRef_(arr),
        sizes_(sizes),
        strides_(strides),
        device_type_(device_type),
        device_idx_(device_idx) {
    assert(sizes.size() == strides.size());
    assert(is_contiguous_strides_for_shape(
        sizes.size(), strides.data(), sizes.data()));
  }

  AtenTensorHandle expensiveCopyToTensor() const {
    AtenTensorHandle result = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
        sizes_.size(),
        sizes_.data(),
        strides_.data(),
        aoti_torch_dtype<std::remove_const_t<T>>(),
        device_type_,
        device_idx_,
        &result));
    void* dataPtr = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(result, &dataPtr));
    std::memcpy(dataPtr, data(), numel() * sizeof(T));
    return result;
  }

  // We need to look the same as RAIIAtenTensorHandle, which returns
  // an owning AtenTensorHandle from release(). So, we allocate one!
  AtenTensorHandle release() {
    return expensiveCopyToTensor();
  }

  AtenTensorHandle borrowAsTensor() const {
    AtenTensorHandle result = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_create_tensor_from_blob_v2(
        data(),
        sizes_.size(),
        sizes_.data(),
        strides_.data(),
        0,
        aoti_torch_dtype<std::remove_const_t<T>>(),
        device_type_,
        device_idx_,
        &result,
        aoti_torch_layout_strided(),
        nullptr,
        0));
    return result;
  }

  // We don't need to free any memory.
  void reset() {}

  auto sizes() const {
    return sizes_;
  }

  auto strides() const {
    return strides_;
  }

  auto device_type() const {
    return device_type_;
  }

  auto device_idx() const {
    return device_idx_;
  }

  T* data() const {
    return arrayRef_.data();
  }

  auto numel() const {
    return arrayRef_.size();
  }

  void set_arrayref(MiniArrayRef<T> new_arrayref) {
    arrayRef_ = new_arrayref;
  }

 private:
  MiniArrayRef<T> arrayRef_;
  // We expect generated code to have statically available sizes &
  // strides for us.
  MiniArrayRef<const int64_t> sizes_;
  MiniArrayRef<const int64_t> strides_;
  int32_t device_type_ = 0;
  int32_t device_idx_ = 0;
  // We continue to zero-initialize this field in case we repurpose
  // the space later; having predictable contents can only help.
  int32_t unusedDoNotRemoveForABICompatibility_ = 0;
};

static_assert(
    sizeof(ArrayRefTensor<int>) ==
        3 * sizeof(MiniIntArrayRef) + 3 * sizeof(int32_t) +
            (alignof(ArrayRefTensor<int>) > 4 ? sizeof(int32_t) : 0),
    "changing the size of ArrayRefTensor breaks ABI compatibility!");

template <typename T>
inline ArrayRefTensor<T> reinterpret_tensor_wrapper(
    const ArrayRefTensor<T>& self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset) {
  // REVIEW: we should add a way to build the DSO in debug mode during
  // tests so we can have checks like this!
  assert(is_contiguous_strides_for_shape(ndim, strides_ptr, sizes_ptr));
  return ArrayRefTensor<T>(
      MiniArrayRef<T>(
          self.data() + storage_offset, self.numel() - storage_offset),
      MiniArrayRef<const int64_t>(sizes_ptr, ndim),
      MiniArrayRef<const int64_t>(strides_ptr, ndim),
      self.device_type(),
      self.device_idx());
}

template <typename T>
inline T* get_data_ptr_wrapper(ArrayRefTensor<T>& tensor) {
  return tensor.data();
}

template <typename T>
inline T* get_data_ptr_wrapper(const MiniArrayRef<T>& arr) {
  return arr.data();
}

template <typename T>
inline const ArrayRefTensor<T>& unwrap_raii_handle_if_needed(
    const ArrayRefTensor<T>& tensor) {
  return tensor;
}

template <typename T>
inline ArrayRefTensor<T>& unwrap_raii_handle_if_needed(
    ArrayRefTensor<T>& tensor) {
  return tensor;
}

template <typename T>
inline const ArrayRefTensor<T>& wrap_with_raii_handle_if_needed(
    const ArrayRefTensor<T>& tensor) {
  return tensor;
}

template <typename T>
inline ArrayRefTensor<T>& wrap_with_raii_handle_if_needed(
    ArrayRefTensor<T>& tensor) {
  return tensor;
}

template <typename T>
inline ArrayRefTensor<T> wrap_with_raii_handle_if_needed(
    ArrayRefTensor<T>&& tensor) {
  return std::move(tensor);
}

template <typename T>
inline RAIIAtenTensorHandle expensive_copy_to_tensor_if_needed(
    const ArrayRefTensor<T>& tensor) {
  return tensor.expensiveCopyToTensor();
}

inline AtenTensorHandle expensive_copy_to_tensor_if_needed(
    AtenTensorHandle handle) {
  return handle;
}

template <typename T>
const T& copy_arrayref_tensor_to_tensor(const T& t) {
  return t;
}

template <typename T>
RAIIAtenTensorHandle copy_arrayref_tensor_to_tensor(
    const ArrayRefTensor<T>& art) {
  return art.expensiveCopyToTensor();
}

template <typename T>
const T& borrow_arrayref_tensor_as_tensor(const T& t) {
  return t;
}

template <typename T>
RAIIAtenTensorHandle borrow_arrayref_tensor_as_tensor(
    const ArrayRefTensor<T>& art) {
  return art.borrowAsTensor();
}

} // namespace torch::aot_inductor
