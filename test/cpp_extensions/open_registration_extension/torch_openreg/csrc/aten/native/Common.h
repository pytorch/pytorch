#include <ATen/EmptyTensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/blob.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/ops/_local_scalar_dense_native.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/quantize_per_tensor_native.h>
#include <ATen/ops/resize_as_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/view_native.h>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function_hook.h>

#include <c10/core/Allocator.h>

#include <include/openreg.h>

namespace at::native::openreg {

class MemoryGuard {
 public:
  template <typename... Args>
  explicit MemoryGuard(const Args&... args) {
    (find_and_unprotect_tensors(args), ...);
  }

  ~MemoryGuard() noexcept {
    for (void* ptr : unprotected_pointers_) {
      orMemoryProtect(ptr);
    }
  }

  MemoryGuard(const MemoryGuard&) = delete;
  MemoryGuard& operator=(const MemoryGuard&) = delete;
  MemoryGuard(MemoryGuard&&) = delete;
  MemoryGuard& operator=(MemoryGuard&&) = delete;

 private:
  template <typename T>
  void find_and_unprotect_tensors(const T& item) {
    if constexpr (std::is_base_of_v<at::TensorBase, T>) {
      unprotect_if_needed(item);
    } else if constexpr (std::is_same_v<T, c10::IValue>) {
      if (item.isTensor()) {
        unprotect_if_needed(item.toTensor());
      } else if (item.isTensorList()) {
        for (const at::Tensor& tensor : item.toTensorListRef()) {
          unprotect_if_needed(tensor);
        }
      } else if (item.isList()) {
        for (const c10::IValue& element : item.toListRef()) {
          find_and_unprotect_tensors(element);
        }
      } else if (item.isGenericDict()) {
        for (const auto& [key, value] : item.toGenericDict()) {
          find_and_unprotect_tensors(key);
          find_and_unprotect_tensors(value);
        }
      }
    }
  }

  void unprotect_if_needed(const at::TensorBase& tensor) {
    if (!tensor.defined() || !tensor.has_storage()) {
      return;
    }

    void* ptr = tensor.data_ptr();
    orPointerAttributes attr;

    if (orPointerGetAttributes(&attr, ptr) != orSuccess ||
        attr.type != orMemoryTypeDevice) {
      return;
    }

    auto [it, inserted] = unprotected_pointers_.insert(attr.pointer);
    if (inserted) {
      orMemoryUnprotect(attr.pointer);
    }
  }

  std::unordered_set<void*> unprotected_pointers_;
};

} // namespace at::native::openreg
