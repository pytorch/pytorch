// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <ATen/functorch/Macros.h>
#include <ATen/Tensor.h>
#include <ATen/functorch/Interpreter.h>

namespace at {
namespace functorch {

// NOTE: [functorch's TensorWrapper]
//
// Taking better suggestions for a name. TensorWrapper is the wrapper Tensor
// Subclass for functorch's grad-based transforms (grad, vjp, jvp). It is
// analogous to how vmap uses BatchedTensor as the wrapper Tensor subclass.
//
// If you're familiar with the Tensor-Variable merge, TensorWrapper is effectively
// another Variable.
//
// Consider grad(grad(torch.sin))(x). This wraps `x` as TensorWrapper(TensorWrapper(x)).
// The reason why is so that each TensorWrapper can hold its own AutogradMeta and
// participate in a **separate** autograd graph.
//
// There are alternative designs we could have chosen (e.g. each grad transform
// stores a weak map of Tensor -> AutogradMeta); the benefit of the TensorWrapper
// design is that we can re-use existing VariableType kernels (i.e. Autograd kernels)
// without much modification. Since a TensorWrapper looks like a regular Tensor,
// the VariableType kernel can pull out the AutogradMeta struct from where it
// expects and extend the autograd graph

struct TORCH_API TensorWrapper : public c10::TensorImpl {
  explicit TensorWrapper(
      c10::DispatchKeySet key_set,
      Tensor value,
      int64_t level,
      std::shared_ptr<bool> is_alive,
      bool is_immutable = false,  // if true, this came from an operation that aliases an immutable tensor
      bool use_value_sizes_strides = true);

  // Override a bunch of methods inherited from TensorImpl to return error messages
  void set_size(int64_t dim, int64_t new_size) override;
  void set_stride(int64_t dim, int64_t new_stride) override;
  void set_storage_offset(int64_t storage_offset) override;

  void refreshMetadata();

  const Tensor& value() const {
    return value_;
  }
  optional<int64_t> level() const {
    if (is_alive()) {
      return level_;
    }
    return {};
  }
  bool is_immutable() const {
    return is_immutable_;
  }
  bool is_alive() const;

  // Overrides necessary for autograd
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const override;
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override;
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;

 private:
  const char* tensorimpl_type_name() const override;
  Tensor value_;
  int64_t level_;
  bool is_immutable_;

  // TensorWrapper receives a boolean flag on whether or not the Grad Interpreter
  // that created it is still alive or not.
  // If the Grad Interpreter is no longer alive then it attempts to behave like
  // a regular Tensor.
  //
  // When we exit the level, this wrapper may be marked as "not alive".
  // Wrappers that are not alive:
  // 1) May still have autograd metadata on them
  // 2) Forward dispatches to the underlying value()
  std::shared_ptr<bool> is_alive_;
};

// There are two variants of makeTensorWrapper: one that accepts a level
// and one that accepts an Interpreter.
//
// The one that accepts a level tries to automatically get the life handle from the
// interpreter on the DynamicLayerStack.
// It needs to be used with caution: if the interpreter is not on the
// DynamicLayerStack, then we won't be able to find the life handle.
//
// In practice this isn't a problem: when we're constructing TensorWrapper in
// Python, the corresponding interpreter is on the stack.
TORCH_API Tensor makeTensorWrapper(const Tensor& tensor, int64_t level, bool is_immutable=false);
TORCH_API Tensor makeTensorWrapper(const Tensor& tensor, const Interpreter& interpreter, bool is_immutable=false);
TORCH_API TensorWrapper* maybeGetTensorWrapper(const Tensor& tensor);
TORCH_API void dumpTensor(std::ostream & ss, const Tensor& tensor);
TORCH_API void dumpTensorCout(const Tensor& tensor);
}
} // namespace at
