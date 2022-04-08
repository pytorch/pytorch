
#include <ATen/FunctionalInverses.h>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
namespace at {
namespace functionalization {

// This logic is similar to autograd code for view backwards calls.
// We can't easily share it though, because (eventually) these functions
// will all call `permute/unsqueeze_copy()` instead of `permute/unsqueeze`.

Tensor permute_copy_inverse(const Tensor& self, IntArrayRef dims) {
  // invert the permutation
  auto ndims = dims.size();
  std::vector<int64_t> dims_(ndims);
  for(const auto i : c10::irange(ndims)) {
    dims_[at::maybe_wrap_dim(dims[i], ndims)] = i;
  }
  return at::permute_copy(self, dims_);
}

Tensor unsqueeze_copy_to(const Tensor & self, IntArrayRef sizes) {
  auto result = self;

  int64_t nDims = sizes.size();
  for(const auto dim : c10::irange(nDims)) {
    if (sizes[dim] == 1) {
      result = at::unsqueeze_copy(result, dim);
    }
  }
  return result;
}

Tensor unsqueeze_copy_to(const Tensor & self, int64_t dim, IntArrayRef sizes) {
  dim = at::maybe_wrap_dim(dim, sizes.size());
  // in NumPy it's not an error to unsqueeze a scalar, but we still need to avoided
  // unsqueezing in the backward.
  if (sizes.size() > 0 && sizes[dim] == 1) {
    return at::unsqueeze_copy(self, dim);
  }
  return self;
}

// Note [Functionalization Pass: View Inverses].
// This file contains the implementation of each "view inverse".
// These aren't really true inverses in the mathematically sense: each view inverse describes how to undo
// the original view (although it takes in different arguments).
//
// E.g. Below is an example of a program that has alias operations removed, and the role that view inverses play:
//
// normal program with views and mutations:
// view1 = input1.view_op(args...)
// view1.add_(1) (perform a mutation on the view, which should also modify input)

// version of the program with no aliasing, that instead uses view_inverse functions:
// view_copy1 = input1.view_copy_op(args...)
// view_copy1.add_(1) (perform a mutation on view_copy1. At this point, input1 is NOT modified)
// x = view_op_inverse(input1, view_copy1, args...)
//
// at this point, input1 and x should be equal
//
// Note that input1 is also passed as an argument to view_op_inverse in the above example.
// This isn't actually required for most view operators: it's only required for view ops
// where you can't figure out what the size of the base tensor is given just the view tensor and arguments.
// Examples are slice/select/scatter/squeeze/as_strided.
// We happen to be passing in the base tensor in all cases, mostly to make the codegen simpler.
// But you'll see below that the "base" argument is ignored by most view_inverse implementations.

// ----------------------------------------------------------
// Implementations of each view_inverse() function are below.
// One of these needs to be implemented for every existing non-composite view operator.
// The codegen automatically generates the corresponding function declaration.
// ----------------------------------------------------------

Tensor FunctionalInverses::_fw_primal_copy_inverse(const at::Tensor& base, const at::Tensor& mutated_view, int64_t level) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call _fw_primal() during the functionalization pass. For now, this is not supported.");
    return Tensor();
}

Tensor FunctionalInverses::_make_dual_copy_inverse(const at::Tensor& base, const at::Tensor& mutated_view, const at::Tensor& tangent, int64_t level) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call _make_dual() during the functionalization pass. For now, this is not supported.");
    return Tensor();
}

Tensor FunctionalInverses::view_as_real_copy_inverse(const Tensor& base, const Tensor& mutated_view) {
    return at::view_as_complex_copy(mutated_view);
}

Tensor FunctionalInverses::view_as_complex_copy_inverse(const Tensor& base, const Tensor& mutated_view) {
    return at::view_as_real_copy(mutated_view.resolve_conj());
}

Tensor FunctionalInverses::_conj_copy_inverse(const Tensor& base, const Tensor& mutated_view) {
    return at::_conj_copy(mutated_view);
}

Tensor FunctionalInverses::_neg_view_copy_inverse(const Tensor& base, const Tensor& mutated_view) {
    return at::_neg_view_copy(mutated_view);
}

Tensor FunctionalInverses::as_strided_copy_inverse(const Tensor& base, const Tensor& mutated_view, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
    TORCH_INTERNAL_ASSERT(false, "as_strided has not been implemented in the functionalization pass yet");
    return Tensor();
}

Tensor FunctionalInverses::diagonal_copy_inverse(const Tensor& base, const Tensor& mutated_view, int64_t offset, int64_t dim1, int64_t dim2) {
    return base.diagonal_scatter(mutated_view, offset, dim1, dim2);
}

Tensor FunctionalInverses::expand_copy_inverse(const Tensor& base, const Tensor& mutated_view, at::IntArrayRef size, bool implicit) {
    return at::sum_to(mutated_view, base.sizes());
}

Tensor FunctionalInverses::permute_copy_inverse(const Tensor& base, const Tensor& mutated_view, at::IntArrayRef dims) {
    return at::functionalization::permute_copy_inverse(mutated_view, dims);
}

Tensor FunctionalInverses::_reshape_alias_copy_inverse(const Tensor& base, const Tensor& mutated_view, at::IntArrayRef size, at::IntArrayRef stride) {
    // Note that I'm directly calling reshape(), and ignoring the strides.
    // _reshape_alias() isn't available from user code, and is an implementation detail of reshape().
    // Specifically, passing in the strides directly can get us into trouble in cases like:
    // b = a[0]; c = b.reshape(...); c.add_(1); print(a)
    // When we eventually run the _reshape_alias_inverse() call here, if we were to pass in both sizes and strides,
    // The call would fail because `mutated_view` doesn't have enough bytes of storage.
    return at::_reshape_alias_copy(mutated_view, base.sizes(), base.strides());
}

Tensor FunctionalInverses::select_copy_int_inverse(const Tensor& base, const Tensor& mutated_view, int64_t dim, int64_t index) {
    return base.select_scatter(mutated_view, dim, index);
}
Tensor FunctionalInverses::detach_copy_inverse(const Tensor& base, const Tensor& mutated_view) {
    // the functionalization pass doesn't care about autograd metadata - as a view, I think detach() is just an identity function
    return mutated_view;
}

Tensor FunctionalInverses::slice_copy_Tensor_inverse(const Tensor& base, const Tensor& mutated_view, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step) {
    return base.slice_scatter(mutated_view, dim, start, end, step);
}

Tensor FunctionalInverses::split_copy_Tensor_inverse(const Tensor& base, const Tensor& mutated_view, int64_t mutated_view_idx, int64_t split_size, int64_t dim) {
    // It would be nice if this logic could be re-used from autograd's split_backward(), but I don't think it can.
    // For functionalization, we have only have one of the tensors from the TensorList outputed by split(), and we want to layer i
    // on top of the base tensor.
    // For autograd, we have all of the tensors outputted by split() and we just want to stack them.
    dim = at::maybe_wrap_dim(dim, base.sizes().size());
    auto dim_size = base.size(dim);
    auto start = mutated_view_idx * split_size;
    auto end = start + split_size;
    if (end > dim_size) end = dim_size;
    return base.slice_scatter(mutated_view, dim, start, end, 1);
}

Tensor FunctionalInverses::split_with_sizes_copy_inverse(const Tensor& base, const Tensor& mutated_view, int64_t mutated_view_idx, at::IntArrayRef split_sizes, int64_t dim) {
    dim = at::maybe_wrap_dim(dim, base.sizes().size());
    auto dim_size = base.size(dim);
    int64_t start = 0;
    for (auto i = 0; i < mutated_view_idx; ++i) {
        start += split_sizes[i];
    }
    auto end = start + split_sizes[mutated_view_idx];
    if (end > dim_size) end = dim_size;
    return base.slice_scatter(mutated_view, dim, start, end, 1);
}

Tensor FunctionalInverses::squeeze_copy_inverse(const Tensor& base, const Tensor& mutated_view) {
    return unsqueeze_copy_to(mutated_view, base.sizes());
}

Tensor FunctionalInverses::squeeze_copy_dim_inverse(const Tensor& base, const Tensor& mutated_view, int64_t dim) {
    return unsqueeze_copy_to(mutated_view, dim, base.sizes());
}

Tensor FunctionalInverses::t_copy_inverse(const Tensor& base, const Tensor& mutated_view) {
    return at::t_copy(mutated_view);
}

Tensor FunctionalInverses::transpose_copy_int_inverse(const Tensor& base, const Tensor& mutated_view, int64_t dim0, int64_t dim1) {
    return transpose_copy(mutated_view, dim0, dim1);
}

Tensor FunctionalInverses::unsqueeze_copy_inverse(const Tensor& base, const Tensor& mutated_view, int64_t dim) {
    return at::squeeze_copy(mutated_view, dim);
}

Tensor FunctionalInverses::_indices_copy_inverse(const Tensor& base, const Tensor& mutated_view) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call _indices() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
    return Tensor();
}

Tensor FunctionalInverses::_values_copy_inverse(const Tensor& base, const Tensor& mutated_view) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call _values() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
    return Tensor();
}

Tensor FunctionalInverses::indices_copy_inverse(const Tensor& base, const Tensor& mutated_view) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call indices() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
    return Tensor();
}

Tensor FunctionalInverses::values_copy_inverse(const Tensor& base, const Tensor& mutated_view) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call values() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
    return Tensor();
}

Tensor FunctionalInverses::_sparse_broadcast_to_copy_inverse(const Tensor& base, const Tensor& mutated_view, at::IntArrayRef size) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call _sparse_broadcast_to() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
    return Tensor();
}

Tensor FunctionalInverses::crow_indices_copy_inverse(const at::Tensor& base, const at::Tensor& mutated_view) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call crow_indices() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
    return Tensor();
}

Tensor FunctionalInverses::col_indices_copy_inverse(const at::Tensor& base, const at::Tensor& mutated_view) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call col_indices() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
    return Tensor();
}

Tensor FunctionalInverses::unbind_copy_int_inverse(const Tensor& base, const Tensor& mutated_view, int64_t mutated_view_idx, int64_t dim) {
    dim = at::maybe_wrap_dim(dim, base.sizes().size());
    return base.select_scatter(mutated_view, dim, mutated_view_idx);
}

Tensor FunctionalInverses::view_copy_inverse(const Tensor& base, const Tensor& mutated_view, at::IntArrayRef size) {
    return at::view_copy(mutated_view, base.sizes());
}

Tensor FunctionalInverses::view_copy_dtype_inverse(const Tensor& base, const Tensor& mutated_view, at::ScalarType dtype) {
    return at::view_copy(mutated_view, base.scalar_type());
}

Tensor FunctionalInverses::unfold_copy_inverse(const Tensor& base, const Tensor& mutated_view, int64_t dimension, int64_t size, int64_t step) {
    // I think autograd and the functionalization pass want the exact same thing here, but need to test to confirm.
    // unfold_backward() is safe to use here because it is NOT a view op.
    return unfold_backward(mutated_view, base.sizes(), dimension, size, step);
}

Tensor FunctionalInverses::alias_copy_inverse(const Tensor& base, const Tensor& mutated_view) {
    return at::alias_copy(mutated_view);
}

} // functionalization
} // at
