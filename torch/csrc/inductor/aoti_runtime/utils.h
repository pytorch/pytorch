#pragma once

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/shim_utils.h>

#if defined(__GNUC__) || defined(__clang__)
#define AOTI_NOINLINE __attribute__((noinline))
#elif _MSC_VER
#define AOTI_NOINLINE __declspec(noinline)
#else
#define AOTI_NOINLINE
#endif

#define AOTI_TORCH_ERROR_CODE_CHECK(call)                                  \
  if ((call) != AOTI_TORCH_SUCCESS) {                                      \
    torch::headeronly::detail::throw_exception(#call, __FILE__, __LINE__); \
  }

using AOTIRuntimeError = int32_t;
#define AOTI_RUNTIME_SUCCESS 0
#define AOTI_RUNTIME_FAILURE 1

#define AOTI_RUNTIME_ERROR_CODE_CHECK(call)                                \
  if ((call) != AOTI_RUNTIME_SUCCESS) {                                    \
    torch::headeronly::detail::throw_exception(#call, __FILE__, __LINE__); \
  }

namespace torch::aot_inductor {

using DeleterFnPtr = void (*)(void*);

inline void noop_deleter(void*) {}

inline void delete_record_function_object(void* ptr) {
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_record_function_end(
      reinterpret_cast<AtenRecordFunctionHandle>(ptr)));
}

inline void delete_tensor_object(void* ptr) {
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_delete_tensor_object(reinterpret_cast<AtenTensorHandle>(ptr)));
}

inline void delete_c10_value_object(void* ptr) {
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_delete_c10_value_object(
      reinterpret_cast<C10IValueHandle>(ptr)));
}

class RAIIAtenRecordFunctionHandle {
 public:
  RAIIAtenRecordFunctionHandle() : handle_(nullptr, noop_deleter) {}
  RAIIAtenRecordFunctionHandle(const RAIIAtenRecordFunctionHandle& other) =
      delete;
  RAIIAtenRecordFunctionHandle& operator=(
      const RAIIAtenRecordFunctionHandle& other) = delete;

  // Initiate an RAII RecordFunction without Inputs
  RAIIAtenRecordFunctionHandle(const char* name, IValueMapHandle kwargs)
      : handle_(nullptr, delete_record_function_object) {
    AtenRecordFunctionHandle tmp_handle = nullptr;
    aoti_record_function_start(name, kwargs, nullptr, 0, &tmp_handle);
    handle_.reset(tmp_handle);
  }

  // Initiate an RAII RecordFunction with Inputs
  RAIIAtenRecordFunctionHandle(
      const char* name,
      IValueMapHandle kwargs,
      std::vector<C10IValueHandle> inputs)
      : handle_(nullptr, delete_record_function_object) {
    AtenRecordFunctionHandle tmp_handle = nullptr;
    aoti_record_function_start(
        name, kwargs, inputs.data(), inputs.size(), &tmp_handle);
    handle_.reset(tmp_handle);
  }

  // Steal the ownership from another RAIIAtenRecordFunctionHandle using
  // std::move
  RAIIAtenRecordFunctionHandle(RAIIAtenRecordFunctionHandle&& other) = default;
  RAIIAtenRecordFunctionHandle& operator=(
      RAIIAtenRecordFunctionHandle&& other) = default;

  // Steal the ownership from raw AtenRecordFunctionHandle
  RAIIAtenRecordFunctionHandle(AtenRecordFunctionHandle handle)
      : handle_(handle, delete_record_function_object) {}

  ~RAIIAtenRecordFunctionHandle() {
    handle_.reset();
  }

  // Return a raw AtenRecordFunctionHandle to be used by aoti_torch functions
  // Note: this function does NOT transfer the ownership of the handle
  operator AtenRecordFunctionHandle() const {
    return handle_.get();
  }

  AtenRecordFunctionHandle release() {
    return handle_.release();
  }

  AtenRecordFunctionHandle get() const {
    return handle_.get();
  }

  void reset() {
    handle_.reset();
  }

 private:
  std::unique_ptr<AtenRecordFunctionOpaque, DeleterFnPtr> handle_;
};

// RAIIAtenTensorHandle steals the tensor objects created by the libtorch C ABI
class RAIIAtenTensorHandle {
 public:
  RAIIAtenTensorHandle() : handle_(nullptr, noop_deleter) {}
  RAIIAtenTensorHandle(const RAIIAtenTensorHandle& other) = delete;
  RAIIAtenTensorHandle& operator=(const RAIIAtenTensorHandle& other) = delete;

  // Steal the ownership from another RAIIAtenTensorHandle using std::move
  RAIIAtenTensorHandle(RAIIAtenTensorHandle&& other) = default;
  RAIIAtenTensorHandle& operator=(RAIIAtenTensorHandle&& other) = default;

  // Steal the ownership from raw AtenTensorHandle
  RAIIAtenTensorHandle(AtenTensorHandle handle)
      : handle_(handle, delete_tensor_object) {}

  ~RAIIAtenTensorHandle() {
    handle_.reset();
  }

  // Return a raw AtenTensorHandle to be used by aoti_torch functions
  // Note: this function does NOT transfer the ownership of the handle
  operator AtenTensorHandle() const {
    return handle_.get();
  }

  AtenTensorHandle release() {
    return handle_.release();
  }

  AtenTensorHandle get() const {
    return handle_.get();
  }

  void reset() {
    handle_.reset();
  }

  int64_t size(int64_t d) {
    int64_t size = 0;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_size(handle_.get(), d, &size));
    return size;
  }

  int64_t stride(int64_t d) {
    int64_t stride = 0;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_stride(handle_.get(), d, &stride));
    return stride;
  }

  int64_t storage_offset() {
    int64_t storage_offset = 0;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_storage_offset(handle_.get(), &storage_offset));
    return storage_offset;
  }

  void* data_ptr() const {
    void* result = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_data_ptr(handle_.get(), &result));
    return result;
  }

  int64_t* sizes() const {
    int64_t* result = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(handle_.get(), &result));
    return result;
  }

  int64_t* strides() const {
    int64_t* result = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(handle_.get(), &result));
    return result;
  }

 private:
  std::unique_ptr<AtenTensorOpaque, DeleterFnPtr> handle_;
};

// RAIIC10IValueHandle steals the IValue objects created by the libtorch C ABI
class RAIIC10IValueHandle {
 public:
  RAIIC10IValueHandle() : handle_(nullptr, noop_deleter) {}
  RAIIC10IValueHandle(const RAIIC10IValueHandle& other) = delete;
  RAIIC10IValueHandle& operator=(const RAIIC10IValueHandle& other) = delete;

  // Steal the ownership from another RAIIC10IValueHandle using std::move
  RAIIC10IValueHandle(RAIIC10IValueHandle&& other) = default;
  RAIIC10IValueHandle& operator=(RAIIC10IValueHandle&& other) = default;

  // Steal the ownership from raw C10IValueHandle
  RAIIC10IValueHandle(C10IValueHandle handle)
      : handle_(handle, delete_c10_value_object) {}

  ~RAIIC10IValueHandle() {
    handle_.reset();
  }

  // Return a raw C10IValueHandle to be used by aoti_torch functions
  // Note: this function does NOT transfer the ownership of the handle
  operator C10IValueHandle() const {
    return handle_.get();
  }

  C10IValueHandle release() {
    return handle_.release();
  }

  C10IValueHandle get() const {
    return handle_.get();
  }

  void reset() {
    handle_.reset();
  }

 private:
  std::unique_ptr<C10IValueOpaque, DeleterFnPtr> handle_;
};

class MaybeOwningAtenTensorHandle {
 public:
  MaybeOwningAtenTensorHandle() : handle_(nullptr) {}
  // We skip copy constructor as MaybeOwningAtenTensorHandle might be RAII which
  // makes it undefined.
  MaybeOwningAtenTensorHandle(const MaybeOwningAtenTensorHandle& other) =
      delete;
  MaybeOwningAtenTensorHandle& operator=(
      const MaybeOwningAtenTensorHandle& other) = delete;

  // Move constructor and move assignment operator
  MaybeOwningAtenTensorHandle(MaybeOwningAtenTensorHandle&& other) = default;
  MaybeOwningAtenTensorHandle& operator=(MaybeOwningAtenTensorHandle&& other) =
      default;

  // Steal the ownership from another RAIIAtenTensorHandle using std::move
  MaybeOwningAtenTensorHandle(RAIIAtenTensorHandle&& other)
      : raii_handle_(std::move(other)) {
    handle_ = raii_handle_.get();
  }
  MaybeOwningAtenTensorHandle& operator=(RAIIAtenTensorHandle&& other) {
    raii_handle_ = std::move(other);
    handle_ = raii_handle_.get();
    return *this;
  }

  // By default, steal the ownership from raw AtenTensorHandle
  MaybeOwningAtenTensorHandle(AtenTensorHandle handle) : raii_handle_(handle) {
    handle_ = raii_handle_.get();
  }

  // If user_managed is true, we do not steal the ownership.
  MaybeOwningAtenTensorHandle(AtenTensorHandle handle, bool user_managed) {
    if (user_managed) {
      aoti_torch_new_tensor_handle(handle, &handle_);
    } else {
      raii_handle_ = RAIIAtenTensorHandle(handle);
      handle_ = raii_handle_.get();
    }
  }

  ~MaybeOwningAtenTensorHandle() {
    // This is no-op if we don't hold raii_handle with the
    // MaybeOwningAtenTensorHandle.
    raii_handle_.reset();
  }

  // Return a raw AtenTensorHandle to be used by aoti_torch functions
  // Note: this function does NOT transfer the ownership of the handle
  operator AtenTensorHandle() const {
    return handle_;
  }

  AtenTensorHandle release() {
    if (raii_handle_) {
      return raii_handle_.release();
    } else {
      AtenTensorHandle handle = handle_;
      handle_ = nullptr;
      return handle;
    }
  }

  AtenTensorHandle get() const {
    return handle_;
  }

  void reset() {
    handle_ = nullptr;
    raii_handle_.reset();
  }

  int64_t size(int64_t d) {
    int64_t size = 0;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_size(handle_, d, &size));
    return size;
  }

  int64_t stride(int64_t d) {
    int64_t stride = 0;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_stride(handle_, d, &stride));
    return stride;
  }

  int64_t storage_offset() {
    int64_t storage_offset = 0;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_storage_offset(handle_, &storage_offset));
    return storage_offset;
  }

  void* data_ptr() const {
    void* result = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(handle_, &result));
    return result;
  }

  int64_t* sizes() const {
    int64_t* result = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(handle_, &result));
    return result;
  }

  int64_t* strides() const {
    int64_t* result = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(handle_, &result));
    return result;
  }

 private:
  // handle_ is the underlying AtenTensorHandle of raii_handle_ if raii_handle_
  // exists. Otherwise it would just be the AtenTensorHandle passed in by users.
  AtenTensorHandle handle_;
  RAIIAtenTensorHandle raii_handle_;
};

// Steal the ownership from raw AtenTensorHandle to RAIIAtenTensorHandle
inline std::vector<RAIIAtenTensorHandle> steal_from_raw_handles_to_raii_handles(
    AtenTensorHandle* handles,
    size_t size) {
  std::vector<RAIIAtenTensorHandle> result;
  result.reserve(size);
  for (size_t i = 0; i < size; i++) {
    result.emplace_back(handles[i]);
    handles[i] = nullptr;
  }
  return result;
}

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

inline void* get_data_ptr_wrapper(AtenTensorHandle tensor) {
  void* result = nullptr;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(tensor, &result));
  return result;
}

inline AtenTensorHandle unwrap_raii_handle_if_needed(
    const RAIIAtenTensorHandle& handle) {
  return handle.get();
}

inline RAIIAtenTensorHandle wrap_with_raii_handle_if_needed(
    AtenTensorHandle handle) {
  return RAIIAtenTensorHandle(handle);
}

class ConstantHandle {
 public:
  ConstantHandle() = default;

  explicit ConstantHandle(AtenTensorHandle handle) : handle_(handle) {
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(handle_, &data_));
  }

  operator AtenTensorHandle() const {
    return handle_;
  }

  AtenTensorHandle tensor() const {
    return handle_;
  }

  AtenTensorHandle get() const {
    return handle_;
  }

  void* data_ptr() const {
    return data_;
  }

  int64_t* sizes() const {
    int64_t* result = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(handle_, &result));
    return result;
  }

  int64_t* strides() const {
    int64_t* result = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(handle_, &result));
    return result;
  }

 private:
  AtenTensorHandle handle_{};
  void* data_ = nullptr;
};

inline void* get_data_ptr_wrapper(const ConstantHandle& constant) {
  return constant.data_ptr();
}

inline const ConstantHandle& unwrap_raii_handle_if_needed(
    const ConstantHandle& handle) {
  return handle;
}

// Shouldn't be called.
inline AtenTensorHandle wrap_with_raii_handle_if_needed(
    const ConstantHandle& handle) = delete;

// DANGEROUS.  Do not call unless you explicitly intend to get a reference to a
// temporary value, which will expire at the end of the current expression.
// This should only be called in cases where the C-shim API expects an optional
// input argument (passed by pointer), and a temporary needs to be passed to it.
template <class T>
T& temporary_reference(T&& t) {
  return t;
}

#define CACHE_TORCH_DTYPE(typename) \
  static auto cached_torch_dtype_##typename = aoti_torch_dtype_##typename()

#define CACHE_TORCH_DEVICE(device)                \
  static auto cached_torch_device_type_##device = \
      aoti_torch_device_type_##device()

#define CACHE_TORCH_LAYOUT(layout) \
  static auto cached_torch_layout_##layout = aoti_torch_layout_##layout()

#define CACHE_TORCH_MEMORY_FORMAT(format)           \
  static auto cached_torch_memory_format_##format = \
      aoti_torch_memory_format_##format()

} // namespace torch::aot_inductor
