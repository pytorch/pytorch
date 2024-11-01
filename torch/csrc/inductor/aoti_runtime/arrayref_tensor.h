#pragma once

#include <torch/csrc/inductor/aoti_runtime/utils.h>

#include <cassert>
#include <cstdint>
#include <cstring>

namespace torch::aot_inductor {

// Can't use c10::ArrayRef because it's not truly header-only and
// pulls in other c10 headers. This is (sadly) copy-pasted and
// adapted.
template <typename T>
class MiniArrayRef final {
 public:
  using iterator = T*;
  using const_iterator = const T*;
  using size_type = size_t;
  using value_type = T;

  using reverse_iterator = std::reverse_iterator<iterator>;

 private:
  /// The start of the array, in an external buffer.
  T* Data;

  /// The number of elements.
  size_type Length;

 public:
  /// @name Constructors
  /// @{

  /// Construct an empty MiniArrayRef.
  /* implicit */ constexpr MiniArrayRef() : Data(nullptr), Length(0) {}

  /// Construct an MiniArrayRef from a single element.
  // TODO Make this explicit
  constexpr MiniArrayRef(const T& OneElt) : Data(&OneElt), Length(1) {}

  /// Construct an MiniArrayRef from a pointer and length.
  constexpr MiniArrayRef(T* data, size_t length) : Data(data), Length(length) {}

  /// Construct an MiniArrayRef from a range.
  constexpr MiniArrayRef(T* begin, T* end) : Data(begin), Length(end - begin) {}

  template <
      typename Container,
      typename = std::enable_if_t<std::is_same_v<
          std::remove_const_t<decltype(std::declval<Container>().data())>,
          T*>>>
  /* implicit */ MiniArrayRef(Container& container)
      : Data(container.data()), Length(container.size()) {}

  /// Construct an MiniArrayRef from a std::vector.
  // The enable_if stuff here makes sure that this isn't used for
  // std::vector<bool>, because MiniArrayRef can't work on a std::vector<bool>
  // bitfield.
  template <typename A>
  /* implicit */ MiniArrayRef(const std::vector<T, A>& Vec)
      : Data(Vec.data()), Length(Vec.size()) {
    static_assert(
        !std::is_same_v<T, bool>,
        "MiniArrayRef<bool> cannot be constructed from a std::vector<bool> bitfield.");
  }

  /// Construct an MiniArrayRef from a std::array
  template <size_t N>
  /* implicit */ constexpr MiniArrayRef(std::array<T, N>& Arr)
      : Data(Arr.data()), Length(N) {}

  /// Construct an MiniArrayRef from a C array.
  template <size_t N>
  // NOLINTNEXTLINE(*c-array*)
  /* implicit */ constexpr MiniArrayRef(T (&Arr)[N]) : Data(Arr), Length(N) {}

  // /// Construct an MiniArrayRef from an empty C array.
  /* implicit */ constexpr MiniArrayRef(const volatile void* Arr)
      : Data(nullptr), Length(0) {}

  /// Construct an MiniArrayRef from a std::initializer_list.
  /* implicit */ constexpr MiniArrayRef(const std::initializer_list<T>& Vec)
      : Data(
            std::begin(Vec) == std::end(Vec) ? static_cast<T*>(nullptr)
                                             : std::begin(Vec)),
        Length(Vec.size()) {}

  /// @}
  /// @name Simple Operations
  /// @{

  constexpr iterator begin() const {
    return Data;
  }
  constexpr iterator end() const {
    return Data + Length;
  }

  // These are actually the same as iterator, since MiniArrayRef only
  // gives you const iterators.
  constexpr const_iterator cbegin() const {
    return Data;
  }
  constexpr const_iterator cend() const {
    return Data + Length;
  }

  constexpr reverse_iterator rbegin() const {
    return reverse_iterator(end());
  }
  constexpr reverse_iterator rend() const {
    return reverse_iterator(begin());
  }

  /// empty - Check if the array is empty.
  constexpr bool empty() const {
    return Length == 0;
  }

  constexpr T* data() const {
    return Data;
  }

  /// size - Get the array size.
  constexpr size_t size() const {
    return Length;
  }

  /// equals - Check for element-wise equality.
  constexpr bool equals(MiniArrayRef RHS) const {
    return Length == RHS.Length && std::equal(begin(), end(), RHS.begin());
  }

  /// @}
  /// @name Operator Overloads
  /// @{
  constexpr const T& operator[](size_t Index) const {
    return Data[Index];
  }

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, MiniArrayRef<T>>& operator=(
      U&& Temporary) = delete;

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, MiniArrayRef<T>>& operator=(
      std::initializer_list<U>) = delete;
};

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

inline AtenTensorHandle reinterpret_tensor_wrapper(
    AtenTensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset) {
  AtenTensorHandle result = nullptr;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__reinterpret_tensor(
      self, ndim, sizes_ptr, strides_ptr, storage_offset, &result));
  return result;
}

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

inline void* get_data_ptr_wrapper(AtenTensorHandle tensor) {
  void* result = nullptr;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(tensor, &result));
  return result;
}

template <typename T>
inline T* get_data_ptr_wrapper(ArrayRefTensor<T>& tensor) {
  return tensor.data();
}

template <typename T>
inline T* get_data_ptr_wrapper(const MiniArrayRef<T>& arr) {
  return arr.data();
}

inline AtenTensorHandle unwrap_raii_handle_if_needed(
    const RAIIAtenTensorHandle& handle) {
  return handle.get();
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

inline RAIIAtenTensorHandle wrap_with_raii_handle_if_needed(
    AtenTensorHandle handle) {
  return RAIIAtenTensorHandle(handle);
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
inline RAIIAtenTensorHandle expensive_copy_to_tensor_if_needed(
    const ArrayRefTensor<T>& tensor) {
  return tensor.expensiveCopyToTensor();
}

inline AtenTensorHandle expensive_copy_to_tensor_if_needed(
    AtenTensorHandle handle) {
  return handle;
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

} // namespace torch::aot_inductor
