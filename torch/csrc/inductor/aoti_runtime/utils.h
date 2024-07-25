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

#if defined(__GNUC__) || defined(__clang__)
#define AOTI_NOINLINE __attribute__((noinline))
#elif _MSC_VER
#define AOTI_NOINLINE __declspec(noinline)
#else
#define AOTI_NOINLINE
#endif

AOTI_NOINLINE static void throw_exception(
    const char* call,
    const char* file,
    int64_t line) {
  std::stringstream ss;
  ss << call << " API call failed at " << file << ", line " << line;
  throw std::runtime_error(ss.str());
}

#define AOTI_TORCH_ERROR_CODE_CHECK(call)       \
  if ((call) != AOTI_TORCH_SUCCESS) {           \
    throw_exception(#call, __FILE__, __LINE__); \
  }

using AOTIRuntimeError = int32_t;
#define AOTI_RUNTIME_SUCCESS 0
#define AOTI_RUNTIME_FAILURE 1

#define AOTI_RUNTIME_ERROR_CODE_CHECK(call)     \
  if ((call) != AOTI_RUNTIME_SUCCESS) {         \
    throw_exception(#call, __FILE__, __LINE__); \
  }

namespace torch::aot_inductor {

using DeleterFnPtr = void (*)(void*);

inline void noop_deleter(void*) {}

inline void delete_tensor_object(void* ptr) {
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_delete_tensor_object(reinterpret_cast<AtenTensorHandle>(ptr)));
}

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
    int64_t size;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_size(handle_.get(), d, &size));
    return size;
  }

  int64_t stride(int64_t d) {
    int64_t stride;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_stride(handle_.get(), d, &stride));
    return stride;
  }

  int64_t storage_offset() {
    int64_t storage_offset;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_storage_offset(handle_.get(), &storage_offset));
    return storage_offset;
  }

 private:
  std::unique_ptr<AtenTensorOpaque, DeleterFnPtr> handle_;
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

 private:
  AtenTensorHandle handle_;
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

#define CACHE_TORCH_DTYPE(typename) \
  static auto cached_torch_dtype_##typename = aoti_torch_dtype_##typename()

#define CACHE_TORCH_DEVICE(device)                \
  static auto cached_torch_device_type_##device = \
      aoti_torch_device_type_##device()

#define CACHE_TORCH_LAYOUT(layout) \
  static auto cached_torch_layout_##layout = aoti_torch_layout_##layout()

} // namespace torch::aot_inductor
