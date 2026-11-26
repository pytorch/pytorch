#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

namespace torch::aot_inductor {

// Functions declared here are not meant to be called from the AOTInductor
// generated model.so

// unsafe_alloc_new_handles_from_tensors is used for allocating new aten
// tensor objects and return them as a vector of AtenTensorHandle (raw
// pointers), and those pointers will be stolen by model.so.
TORCH_API std::vector<AtenTensorHandle> unsafe_alloc_new_handles_from_tensors(
    const std::vector<at::Tensor>& tensors);

// alloc_tensors_by_stealing_from_handles is used for creating a vector of aten
// tensors by stealing from an array of handles. Only the handles are stolen,
// and the array itself is borrowed.
//
// WARNING: Can NOT be called in model.so
TORCH_API std::vector<at::Tensor> alloc_tensors_by_stealing_from_handles(
    AtenTensorHandle* handles,
    size_t length);

// free_unstolen_handles frees any handles a stealing callee did not take. A
// stealing run (the generated model.so, or
// alloc_tensors_by_stealing_from_handles) nulls each slot it takes ownership
// of, so this is safe to run on every exit of a run path: after a fully
// successful, fully stealing run every slot is null and this is a no-op; if the
// run throws before stealing, it releases the un-stolen handles instead of
// leaking their tensor storage.
//
// WARNING: Can NOT be called in model.so
TORCH_API void free_unstolen_handles(std::vector<AtenTensorHandle>& handles);

} // namespace torch::aot_inductor
