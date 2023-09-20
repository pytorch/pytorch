#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <ATen/Tensor.h>

namespace torch {
namespace aot_inductor {

// Functions declared here are not meant to be called from the AOTInductor
// generated model.so

// No ownership transfer, just pointer type conversion
TORCH_API at::Tensor* tensor_handle_to_tensor_pointer(AtenTensorHandle handle);

// No ownership transfer, just pointer type conversion
TORCH_API AtenTensorHandle tensor_pointer_to_tensor_handle(at::Tensor* tensor);

// alloc_tensors_by_stealing_from_handles is used for creating a vector of aten
// tensors by stealing from a vector of handles
// WARNING: only used in the non ABI compatible mode
TORCH_API std::vector<at::Tensor> alloc_tensors_by_stealing_from_handles(
    std::vector<AtenTensorHandle>& handles);

class TORCH_API RAIITensorToHandleAllocator {
 public:
  RAIITensorToHandleAllocator() = delete;
  RAIITensorToHandleAllocator(const RAIITensorToHandleAllocator& other) =
      delete;
  RAIITensorToHandleAllocator(RAIITensorToHandleAllocator&& other) = delete;
  RAIITensorToHandleAllocator& operator=(
      const RAIITensorToHandleAllocator& other) = delete;
  RAIITensorToHandleAllocator& operator=(RAIITensorToHandleAllocator&& other) =
      delete;

  // Allocate new tensor objects and allocate an array to store pointers
  // (AtenTensorHandle) to those objects
  RAIITensorToHandleAllocator(std::vector<at::Tensor>& tensors);

  ~RAIITensorToHandleAllocator() {
    if (handles_) {
      delete[] handles_;
    }
  }

  // Release the handle array and the caller is REQUIRED to steal both the
  // ownership of the allocated array and the allocated tensor objects stored as
  // pointers in that array
  AtenTensorHandle* release() {
    AtenTensorHandle* result = handles_;
    handles_ = nullptr;
    return result;
  }

 private:
  AtenTensorHandle* handles_;
};

} // namespace aot_inductor
} // namespace torch
