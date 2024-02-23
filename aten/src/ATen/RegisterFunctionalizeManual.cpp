#include <ATen/RegisterFunctionalizeManual.h>

#include <ATen/InferSize.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/irange.h>
#include <c10/util/strides.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_to_copy.h>
#include <ATen/ops/_unsafe_view.h>
#include <ATen/ops/as_strided.h>
#include <ATen/ops/as_strided_copy.h>
#include <ATen/ops/empty_strided_native.h>
#include <ATen/ops/lift.h>
#include <ATen/ops/lift_fresh.h>
#include <ATen/ops/lift_fresh_copy.h>
#include <ATen/ops/resize.h>
#include <ATen/ops/to_native.h>

#include <utility>
#endif

using at::functionalization::exclude_keys_for_meta_dispatch;
using at::functionalization::to_meta;

// resize_() is special because:
// - when we resize to a larger size, it acts as a mutation
// - when we resize to a smaller size, it acts as a view
// See Note [resize_ in Functionalization] for more dtails
static const at::Tensor & resize__functionalization(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format) {
  // First unwrap the tensor arguments
  at::Tensor self_;
  if (at::functionalization::impl::isFunctionalTensor(self)) {
    at::functionalization::impl::sync(self);
    self_ = at::functionalization::impl::from_functional_tensor(self);
  } else {
    self_ = self;
  }
  // Case 1: arguments are not functional tensors, so we no-op and redispatch.
  if (!at::functionalization::impl::isFunctionalTensor(self)) {
     at::AutoDispatchSkipFunctionalize guard;
     self_.resize_(size, memory_format);
     return self;
  }

  // Case 2: actually functionalize resize_()
  at::Tensor tmp_output;
  {
    at::AutoDispatchSkipFunctionalize guard;
    tmp_output = at::resize(self_, size, memory_format);
  }

  auto itemsize = self.dtype().itemsize();
  auto storage_offset = self.storage_offset();
  auto new_size_bytes = at::detail::computeStorageNbytesContiguous(size, itemsize, storage_offset);
  auto needs_resize_storage = new_size_bytes > self.storage().nbytes();

  if (needs_resize_storage) {
    // If resize_() actually increases the size of the storage, then we need to tell FunctionalTensorWrapper about it.
    // See Note[resize_() in functionalization pass]
    auto func_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(self);
    func_impl->maybe_replace_storage(tmp_output);
    // See the note - we're guaranteed at this point that "self" is *not* a view (and has no outstanding views)
    // So we don't need to treat the output of resize as view tensor.
    return self;
  }

  // Otherwise, we know that we're resizing to a smaller size.
  // resize_() is effectively a view operator.
  // The output of resizing is equivalent to taking a slice of a larger tensor.
  // We have to emulate this "slicing" with an as_strided call.
  auto reapply_views = at::functionalization::impl::getFunctionalizationReapplyViewsTLS();
  at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(
    [reapply_views = reapply_views, size = size.vec()](const at::Tensor & base, int64_t mutated_view_idx) -> at::Tensor {
      if (reapply_views) {
        return base.as_strided(size, c10::contiguous_strides(size));
      } else {
        return at::as_strided_copy(base, size, c10::contiguous_strides(size));
      }
    },
    [size = size.vec()](const at::Tensor & base, const at::Tensor & mutated_view, int64_t mutated_view_idx) -> at::Tensor {
      return base.as_strided_scatter(mutated_view, size, c10::contiguous_strides(size));
    }
  );
  at::functionalization::impl::mutate_view_meta(self, std::move(view_meta));
  return self;
}


static at::Tensor lift_functionalize(const at::Tensor & self) {
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(self));
  at::AutoDispatchSkipFunctionalize guard;
  auto out = at::lift(self);
  return at::functionalization::impl::to_functional_tensor(out);
}

static at::Tensor lift_fresh_functionalize(const at::Tensor & self) {
  // See Note [Exporting and compiling a graph with lift_fresh_copy]
  if (at::functionalization::impl::isFunctionalTensor(self)) {
    return self.view_as(self);
  }

  at::AutoDispatchSkipFunctionalize guard;
  auto out = at::lift_fresh(self);
  return at::functionalization::impl::to_functional_tensor(out);
}

static at::Tensor lift_fresh_functionalize_copy(const at::Tensor & self) {
  // Note [Exporting and compiling a graph with lift_fresh_copy]
  // If out is already a functional tensor, don't wrap it twice.
  // In theory this could be useful if we want to nest functionalization with itself,
  // but that isn't really a use case today.
  // Needed for https://github.com/pytorch/pytorch/issues/105327
  if (at::functionalization::impl::isFunctionalTensor(self)) {
    return self.clone();
  }

  at::AutoDispatchSkipFunctionalize guard;
  auto out = at::lift_fresh_copy(self);
  return at::functionalization::impl::to_functional_tensor(out);
}

static bool device_opted_into_functionalization(c10::Device self_device, c10::optional<c10::Device> tgt_device) {
    // If the target device is empty, then the output tensor should be on the same device as the input
    auto real_tgt_device = tgt_device.has_value() ? tgt_device.value() : self_device;
    return real_tgt_device.type() == c10::DeviceType::XLA || real_tgt_device.type() == c10::DeviceType::Lazy;
}

// note I only need this because the to.dtype/to.dtype_layout overload calls this, so we skip the op above.
// We should probably get rid of this though.
static at::Tensor _to_copy_functionalize(
        const at::Tensor & self,
        c10::optional<at::ScalarType> dtype,
        c10::optional<at::Layout> layout,
        c10::optional<at::Device> device,
        c10::optional<bool> pin_memory,
        bool non_blocking,
        c10::optional<at::MemoryFormat> memory_format) {
  at::Tensor self_;
  if (at::functionalization::impl::isFunctionalTensor(self)) {
    // sync any pending updates
    at::functionalization::impl::sync(self);
    // pass the unwrapped tensor to the backend
    self_ = at::functionalization::impl::from_functional_tensor(self);
  } else {
    self_ = self;
  }

  at::AutoDispatchSkipFunctionalize guard;
  auto out = at::_to_copy(self_, dtype, layout, device, pin_memory, non_blocking, memory_format);

  // Special case: if the Functionalize key is not in TLS, we assume that we're running
  // on a lazy backend (LTC).
  // In that case, if we're copying to a non-functionalize-enabled device,
  // then the functionalization pass should "end". We need to sync any updates on the input
  // tensor, but we shouldn't wrap the output.
  if (!c10::impl::tls_local_dispatch_key_set().included_.has(c10::DispatchKey::Functionalize)) {
    if (!device_opted_into_functionalization(self.device(), device)) {
      return out;
    }
  }
  return at::functionalization::impl::to_functional_tensor(out);
}

// Why is _unsafe_view special-cased here?
// Basically just to satisfy autograd's debug asserts.
// The situation:
// - _unsafe_view's autograd kernel has debug asserts to confirm
//   that the input and output alias storage.
// - _unsafe_view's schema in native_functions.yaml
//   does not contain alias annotations, so it advertises as non-aliasing.
// - functionalization will then treat _unsafe_view like a non-aliasing op.
//   Specifically, autograd will redispatch to functionalization's
//   boxed fallback kernel, which creates a new FunctionalTensorWrapper output
//   that does **not** alias storage with the input, tripping the assert.
// The kernel written here just manually re-ifies the aliasing relationship.
//
// Another way to handle this would be to fix unsafe_view's alias annotations
// in native_functions.yaml, but I think this would be a pessimization.
// The idea with _unsafe_view is that you're guaranteed that the input
// is a temporary, and don't actually have to worry about propagating
// mutations between the input and output.
static at::Tensor _unsafe_view_functionalize(const at::Tensor & self, at::SymIntArrayRef size) {
  if (!at::functionalization::impl::isFunctionalTensor(self)) {
    at::AutoDispatchSkipFunctionalize guard;
    return at::_unsafe_view_symint(self, size);
  }

  auto self_ = at::functionalization::impl::from_functional_tensor(self);
  at::Tensor tmp_output;
  {
    at::AutoDispatchSkipFunctionalize guard;
    tmp_output = at::_unsafe_view_symint(self_, size);
  }

  at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(
    [size = size.vec()](const at::Tensor & base, int64_t mutated_view_idx) -> at::Tensor {
      return at::_unsafe_view_symint(base, size);
    },
    [size = size.vec()](const at::Tensor & base, const at::Tensor & mutated_view, int64_t mutated_view_idx) -> at::Tensor {
      return at::_unsafe_view_symint(mutated_view, base.sym_sizes());
    }
  );

  auto out = at::functionalization::impl::create_functional_tensor_with_view_meta(tmp_output, self, std::move(view_meta));
  // See  Note [Propagating strides in the functionalization pass]
  // (for _unsafe_view, I'm just manually doing the shape inference rule here instead of calling the meta function for unsafe_view)
  auto inferred_size = at::infer_size_dv(size, self.sym_numel());
  auto stride = at::detail::computeStride(self.sym_sizes(), self.sym_strides(), inferred_size);
  TORCH_INTERNAL_ASSERT(stride.has_value());
  out.unsafeGetTensorImpl()->set_sizes_and_strides(inferred_size, stride.value());
  return out;
}

static at::Tensor& set__functionalize(at::Tensor& self, const at::Tensor& src) {
  // error case
  TORCH_CHECK(at::functionalization::impl::isFunctionalTensor(self) || !at::functionalization::impl::isFunctionalTensor(src),
    "set__functionalize: Tried to mutate a non-functional tensor with a functional tensor, which is not allowed");

  TORCH_CHECK(at::functionalization::impl::isFunctionalTensor(src),
    "set__functionalize: We do not currently support x.set_(y) where y is not a FunctionalTensor. Please file an issue");

  // nop case
  if (!at::functionalization::impl::isFunctionalTensor(self) && !at::functionalization::impl::isFunctionalTensor(src)) {
    at::AutoDispatchSkipFunctionalize guard;
    return self.set_(src);
  }

  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self));
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(src));
  auto self_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(self);
  auto src_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(src);
  self_impl->set__impl(src_impl);
  return self;
}

// [Note: as_strided Functionalization]
//
// 'as_strided' (and its inplace version 'as_strided_') is special-cased because
// contrary to other operations, it cares mostly about the underlying storage,
// instead of the actual tensor. We can see that in the following example:
//
// a = torch.arange(10)
// b = a[::2]
// c = b.as_strided((10,), (1,))
//
// 'c' is only possible because, as mentioned above, 'as_strided' operates on the
// actual storage of 'b', which is the exact same as 'a'. Since the storage of 'a'
// had the capacity for, at least, 10 elements, 'c' is able to also have, at most,
// 10 elements.
//
// Therefore, instead of calling the actual 'as_strided' operation (c) with the given
// arguments (b), we call it with the argument's base tensor (a).

// The two functions below (as_strided_functionalize and as_strided__functionalize)
// are slightly modified versions of their code-generated ones (RegisterFunctionalization_X.cpp).
// Below, we call out each of those changes by a 'CHANGE' flag.
static at::Tensor as_strided_functionalize(
    c10::DispatchKeySet dispatchKeySet,
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<c10::SymInt> storage_offset) {
  at::Tensor self_;

  if (at::functionalization::impl::isFunctionalTensor(self)) {
    self_ = at::functionalization::impl::from_functional_tensor(self);
  } else {
    self_ = self;
  }

  if (!at::functionalization::impl::isFunctionalTensor(self)) {
    // functionalization is re-entrant, but will no-op if it wasn't passed a
    // FunctionalTensorWrapper.
    at::AutoDispatchSkipFunctionalize guard;
    return at::_ops::as_strided::call(self_, size, stride, storage_offset);
  }

  // CHANGE: create a FunctionalTensorWrapper for the base tensor of 'self'.
  auto self_base =
      at::functionalization::impl::create_functional_tensor_from_base(self);
  auto self_base_ =
      at::functionalization::impl::from_functional_tensor(self_base);
  // CHANGE: pass-through the storage offset value from 'self', which is the
  //         only aspect of 'self' that 'as_strided' cares about.
  auto storage_offset_ = storage_offset.value_or(self.sym_storage_offset());

  auto reapply_views =
      at::functionalization::impl::getFunctionalizationReapplyViewsTLS();
  auto inverse_return_mode =
      (reapply_views
           ? at::functionalization::InverseReturnMode::ViewOrScatterInverse
           : at::functionalization::InverseReturnMode::NeverView);
  auto compute_reference_meta =
      self.key_set().has_backend(c10::BackendComponent::XLABit) ||
      self.key_set().has_backend(c10::BackendComponent::LazyBit);
  at::Tensor reference_tensor_output;
  if (compute_reference_meta) {
    // CHANGE: we should also use the base tensor for meta reference.
    auto self_meta = to_meta(self_base);
    at::AutoDispatchSkipFunctionalize func_guard;
    c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);
    reference_tensor_output =
        at::_ops::as_strided::call(self_meta, size, stride, storage_offset_);
  }
  at::Tensor tmp_output;
  {
    // CHANGE: call the actual operation with the base tensor.
    at::AutoDispatchSkipFunctionalize guard;
    if (reapply_views) {
      tmp_output =
          at::_ops::as_strided::call(self_base_, size, stride, storage_offset_);
    } else {
      tmp_output = at::_ops::as_strided_copy::call(
          self_base_, size, stride, storage_offset_);
    }
  }
  at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(
      [reapply_views = reapply_views,
       size = size.vec(),
       stride = stride.vec(),
       storage_offset = storage_offset_](
          const at::Tensor& base, int64_t mutated_view_idx) -> at::Tensor {
        if (reapply_views) {
          return at::_ops::as_strided::call(base, size, stride, storage_offset);
        } else {
          return at::_ops::as_strided_copy::call(
              base, size, stride, storage_offset);
        }
      },
      [inverse_return_mode = inverse_return_mode,
       size = size.vec(),
       stride = stride.vec(),
       storage_offset = storage_offset_](
          const at::Tensor& base,
          const at::Tensor& mutated_view,
          int64_t mutated_view_idx) -> at::Tensor {
        return at::functionalization::FunctionalInverses::as_strided_inverse(
            base,
            mutated_view,
            inverse_return_mode,
            size,
            stride,
            storage_offset);
      },
      /*is_multi_output=*/false);
  // CHANGE: the result FunctionalTensorWrapper should use, as its base the
  //         actual base of the 'self' argument.
  auto out =
      at::functionalization::impl::create_functional_tensor_with_view_meta(
          tmp_output, self_base, view_meta);
  // See  Note [Propagating strides in the functionalization pass]
  if (compute_reference_meta) {
    at::functionalization::impl::set_sizes_strides_offset(
        out, reference_tensor_output);
  }
  return out;
}

static const at::Tensor& as_strided__functionalize(
    c10::DispatchKeySet dispatchKeySet,
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<c10::SymInt> storage_offset) {
  if (!at::functionalization::impl::isFunctionalTensor(self)) {
    // functionalization is re-entrant, but will no-op if it wasn't passed a
    // FunctionalTensorWrapper.
    at::AutoDispatchSkipFunctionalize guard;
    return at::_ops::as_strided_::call(self, size, stride, storage_offset);
  }

  // CHANGE: pass-through the storage offset value from 'self', which is the
  //         only aspect of 'self' that 'as_strided' cares about.
  auto storage_offset_ = storage_offset.value_or(self.sym_storage_offset());

  auto reapply_views =
      at::functionalization::impl::getFunctionalizationReapplyViewsTLS();
  auto inverse_return_mode =
      (reapply_views
           ? at::functionalization::InverseReturnMode::ViewOrScatterInverse
           : at::functionalization::InverseReturnMode::NeverView);

  at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(
      [reapply_views = reapply_views,
       size = size.vec(),
       stride = stride.vec(),
       storage_offset = storage_offset_](
          const at::Tensor& base, int64_t mutated_view_idx) -> at::Tensor {
        if (reapply_views) {
          return at::_ops::as_strided::call(base, size, stride, storage_offset);
        } else {
          return at::_ops::as_strided_copy::call(
              base, size, stride, storage_offset);
        }
      },
      [inverse_return_mode = inverse_return_mode,
       size = size.vec(),
       stride = stride.vec(),
       storage_offset = storage_offset_](
          const at::Tensor& base,
          const at::Tensor& mutated_view,
          int64_t mutated_view_idx) -> at::Tensor {
        return at::functionalization::FunctionalInverses::as_strided_inverse(
            base,
            mutated_view,
            inverse_return_mode,
            size,
            stride,
            storage_offset);
      });

  auto self_base =
      at::functionalization::impl::create_functional_tensor_from_base(self);
  auto compute_reference_meta =
      self.key_set().has_backend(c10::BackendComponent::XLABit) ||
      self.key_set().has_backend(c10::BackendComponent::LazyBit);
  at::Tensor reference_tensor_output;
  if (compute_reference_meta) {
    auto self_meta = to_meta(self_base);
    at::AutoDispatchSkipFunctionalize func_guard;
    c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);
    reference_tensor_output =
        at::_ops::as_strided_::call(self_meta, size, stride, storage_offset_);
  }
  // This function adds the above view meta to the current tensor and replays
  // them off the base, mutating the size/stride info of the current
  // FunctionalTensorWrapper. Because of this, we need to make sure to run the
  // reference shape function above, BEFORE doing this (otherwise we'll end up
  // runnin the reference function using the wrong sizes/strides)

  // CHANGE: apply the new ViewMeta to the created base tensor.
  at::functionalization::impl::mutate_view_meta(self_base, view_meta);

  // See  Note [Propagating strides in the functionalization pass]
  // XLA/LTC don't implement the logic to propagate strides correctly, so we
  // need to rely on a reference implementation here (instead of relying on the
  // output from the forward lambda having the correct stride info)
  if (compute_reference_meta) {
    at::functionalization::impl::set_sizes_strides_offset(
        self_base, reference_tensor_output);
  }

  // CHANGE: replace the underlying TensorImpl to the one we just modified.
  self.set_(self_base);
  return self;
}

TORCH_LIBRARY_IMPL(aten, Functionalize, m) {
  m.impl("resize_", TORCH_FN(resize__functionalization));
  m.impl("lift", TORCH_FN(lift_functionalize));
  m.impl("lift_fresh", TORCH_FN(lift_fresh_functionalize));
  m.impl("lift_fresh_copy", TORCH_FN(lift_fresh_functionalize_copy));
  m.impl("_to_copy", TORCH_FN(_to_copy_functionalize));
  m.impl("_unsafe_view", TORCH_FN(_unsafe_view_functionalize));
  // The overloads of set_() that take in a storage should never
  // appear with torch.compile, because dynamo graph breaks
  m.impl("set_.source_Tensor", TORCH_FN(set__functionalize));
  m.impl("as_strided", TORCH_FN(as_strided_functionalize));
  m.impl("as_strided_", TORCH_FN(as_strided__functionalize));
}
