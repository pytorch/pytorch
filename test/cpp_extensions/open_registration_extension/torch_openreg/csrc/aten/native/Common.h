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

#include <set>

#include <include/openreg.h>

namespace at::native {

class MemoryGuard {
 public:
  explicit MemoryGuard(const torch::jit::Stack& stack) {
    for (const c10::IValue& ivalue : stack) {
      find_and_unprotect_tensors(ivalue);
    }
  }

  template <typename... Args>
  explicit MemoryGuard(const Args&... args) {
    (handler(args), ...);
  }

  ~MemoryGuard() {
    for (void* ptr : unprotected_pointers_) {
      orMemoryProtect(ptr);
    }
  }

  MemoryGuard(const MemoryGuard&) = delete;
  MemoryGuard& operator=(const MemoryGuard&) = delete;
  MemoryGuard(MemoryGuard&&) = delete;
  MemoryGuard& operator=(MemoryGuard&&) = delete;

 private:
  void find_and_unprotect_tensors(const c10::IValue& ivalue) {
    if (ivalue.isTensor()) {
      unprotect_if_needed(ivalue.toTensor());
    } else if (ivalue.isTensorList()) {
      for (const at::Tensor& tensor : ivalue.toTensorList()) {
        unprotect_if_needed(tensor);
      }
    } else if (ivalue.isList()) {
      for (const c10::IValue& element : ivalue.toListRef()) {
        find_and_unprotect_tensors(element);
      }
    } else if (ivalue.isGenericDict()) {
      for (const auto& pair : ivalue.toGenericDict()) {
        find_and_unprotect_tensors(pair.key());
        find_and_unprotect_tensors(pair.value());
      }
    }
  }

  void unprotect_if_needed(const at::Tensor& tensor) {
    if (!tensor.defined() || !tensor.has_storage()) {
      return;
    }

    void* ptr = tensor.data_ptr();
    orPointerAttributes attr;

    if (orPointerGetAttributes(&attr, ptr) == orSuccess) {
      if (attr.type == orMemoryTypeDevice) {
        if (unprotected_pointers_.find(attr.pointer) ==
            unprotected_pointers_.end()) {
          orMemoryUnprotect(attr.pointer);
          unprotected_pointers_.insert(attr.pointer);
        }
      }
    }
  }

  template <typename T>
  void handler(const T& x) {
    if constexpr (std::is_same_v<std::decay_t<T>, at::Tensor>) {
      unprotect_if_needed(x);
    }
  }

  std::set<void*> unprotected_pointers_;
};

} // namespace at::native
