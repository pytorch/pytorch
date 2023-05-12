// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/TensorWrapper.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/functorch/BatchedTensorImpl.h>

#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at {
namespace functorch {

void dumpTensor(std::ostream& ss, const Tensor& tensor) {
  auto* wrapped = maybeGetTensorWrapper(tensor);
  if (!wrapped) {
    auto* batched = maybeGetBatchedImpl(tensor);
    if (batched) {
      ss << "Batched[lvl=" << batched->level() << " dim=" << batched->bdim() << ", ";
      dumpTensor(ss, batched->value());
      ss << "]";
      return;
    }
    ss << "Tensor" << tensor.sizes();
    return;
  }
  ss << "Wrapper[";
  if (wrapped->level().has_value()) {
    ss << "lvl=" << wrapped->level().value() << ", ";
  } else {
    ss << "dead, ";
  }
  dumpTensor(ss, wrapped->value());
  ss << "]";
}

void TensorWrapper::refreshMetadata() {
  auto dim = value_.dim();
  auto sizes = value_.sizes();
  auto strides = value_.strides();
  storage_offset_ = value_.storage_offset();
  sizes_and_strides_.resize(value_.dim());
  for (int64_t i = 0; i < dim; i++) {
    sizes_and_strides_.size_at_unchecked(i) = sizes[i];
    sizes_and_strides_.stride_at_unchecked(i) = strides[i];
  }

  refresh_numel();
  refresh_contiguous();
}

void dumpTensorCout(const Tensor& tensor) {
  dumpTensor(std::cout, tensor);

  std::cout << std::endl;
}

static c10::intrusive_ptr<TensorWrapper> makeTensorWrapperPtr(const Tensor& tensor, int64_t level, const std::shared_ptr<bool>& life_handle) {
  auto keys_to_propagate = kKeysToPropagateToWrapper | DispatchKeySet({
      DispatchKey::AutogradCPU, DispatchKey::AutogradCUDA, DispatchKey::AutogradXLA});
  auto key_set = getKeysToPropagateToWrapper(tensor, keys_to_propagate);
  key_set = key_set.add(DispatchKey::FuncTorchGradWrapper);
  return c10::make_intrusive<TensorWrapper>(key_set, tensor, level, life_handle);
}

// use makeTensorWrapper instead to avoid potential footguns:
// unsafeMakeTensorWrapper doesn't check that level and life_handle
// refer to the same interpreter
static Tensor unsafeMakeTensorWrapper(
    const Tensor& tensor,
    int64_t level,
    bool is_immutable,
    const std::shared_ptr<bool>& life_handle) {
  auto wrapped = maybeGetTensorWrapper(tensor);
  if (wrapped) {
    TORCH_INTERNAL_ASSERT(wrapped->level() < level);
  }

  auto keys_to_propagate = kKeysToPropagateToWrapper | DispatchKeySet({
      DispatchKey::AutogradCPU, DispatchKey::AutogradCUDA, DispatchKey::AutogradXLA});
  auto key_set = getKeysToPropagateToWrapper(tensor, keys_to_propagate);
  key_set = key_set.add(DispatchKey::FuncTorchGradWrapper);
  auto result = at::detail::make_tensor<TensorWrapper>(
      key_set, tensor, level, life_handle, is_immutable);
  TORCH_INTERNAL_ASSERT(result.key_set().has(DispatchKey::FuncTorchGradWrapper));
  return result;
}

Tensor makeTensorWrapper(const Tensor& tensor, int64_t level, bool is_immutable) {
  auto life_handle = getLifeHandleForLevel(level);
  return unsafeMakeTensorWrapper(
      tensor,
      level,
      is_immutable,
      getLifeHandleForLevel(level));
}

Tensor makeTensorWrapper(const Tensor& tensor, const Interpreter& interpreter, bool is_immutable) {
  return unsafeMakeTensorWrapper(
      tensor,
      interpreter.level(),
      is_immutable,
      interpreter.is_alive_ptr());
}


bool TensorWrapper::is_alive() const {
  return *is_alive_;
}

c10::intrusive_ptr<TensorImpl> TensorWrapper::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  auto dest_impl = makeTensorWrapperPtr(value(), level_, is_alive_);
  dest_impl->set_version_counter(version_counter);

  // TODO: is this even right?
  dest_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
  return dest_impl;
}

c10::intrusive_ptr<TensorImpl> TensorWrapper::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  auto dest_impl = makeTensorWrapperPtr(value(), level_, is_alive_);
  dest_impl->set_version_counter(version_counter);

  // TODO: is this even right?
  dest_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
  return dest_impl;
}

void TensorWrapper::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
  TORCH_CHECK(false, "mutating directly with `.data` inside functorch transform is not allowed.");
}

TensorWrapper::TensorWrapper(
    c10::DispatchKeySet key_set,
    Tensor value,
    int64_t level,
    std::shared_ptr<bool> is_alive,
    bool is_immutable,
    bool use_value_sizes_strides)
  : TensorImpl(key_set, value.dtype(), value.device())
  , value_(std::move(value))
  , level_(level)
  , is_immutable_(is_immutable)
  , is_alive_(std::move(is_alive))
{
  TORCH_INTERNAL_ASSERT(value_.defined());

  // TODO: need to reset sizes/strides on mutation
  TORCH_INTERNAL_ASSERT(use_value_sizes_strides);
  refreshMetadata();

  set_storage_access_should_throw();
}

// The following are some internal inherited methods that we do not support.
// They should never get called.
void TensorWrapper::set_size(int64_t dim, int64_t new_size) {
  TORCH_INTERNAL_ASSERT(false, "Can't set_size for TensorWrapper");
}
void TensorWrapper::set_stride(int64_t dim, int64_t new_stride) {
  TORCH_INTERNAL_ASSERT(false, "Can't set_stride for TensorWrapper");
}
void TensorWrapper::set_storage_offset(int64_t storage_offset) {
  TORCH_INTERNAL_ASSERT(false, "Can't set_storage_offset for TensorWrapper");
}

const char* TensorWrapper::tensorimpl_type_name() const {
  return "TensorWrapper";
}


TensorWrapper* maybeGetTensorWrapper(const Tensor& tensor) {
  if (!tensor.key_set().has(DispatchKey::FuncTorchGradWrapper)) {
    return nullptr;
  }
  return (TensorWrapper*)(tensor.unsafeGetTensorImpl());
}

static void dead_tensor_wrapper_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto args_size = op.schema().arguments().size();
  int64_t unwrapped_count = 0;
  auto unwrapIfDeadAndIncrement = [&](const Tensor& tensor) {
    auto* wrapped = maybeGetTensorWrapper(tensor);
    if (!wrapped) {
      return tensor;
    }

    // NOTE: We need to test for both is_alive and functorch mode dispatch keys
    //       being active because certain ops may disable the keys but not set
    //       the relevant tensor's state to dead.
    //       Example: torch.tensor([x, y, z]) - variant which accepts list of scalars
    //       leads to the above case.
    constexpr auto functorch_mode_ks = DispatchKeySet(
        {DispatchKey::FuncTorchDynamicLayerFrontMode,
         DispatchKey::FuncTorchDynamicLayerBackMode});
    if (wrapped->is_alive() && wrapped->key_set().has_any(functorch_mode_ks)) {
      return tensor;
    }
    unwrapped_count++;
    return wrapped->value();
  };

  foreachTensorInplace(*stack, stack->size() - args_size, stack->size(), unwrapIfDeadAndIncrement);
  TORCH_INTERNAL_ASSERT(unwrapped_count > 0, "Should have at least one dead wrapper");

  // re-dispatch
  op.callBoxed(stack);
}

// TensorWrapper backend fallback: Unwrap and fallthrough.

TORCH_LIBRARY_IMPL(_, FuncTorchGradWrapper, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&dead_tensor_wrapper_fallback>());
}

}
} // namespace at
