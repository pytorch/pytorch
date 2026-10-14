
#include <ATen/FunctionalInverses.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ExpandUtils.h>
#include <ATen/WrapDimUtilsMulti.h>

#include <utility>
namespace at::functionalization {

// This logic is similar to autograd code for view backwards calls.
// We can't easily share it though, because (eventually) these functions
// will all call `permute/unsqueeze_copy()` instead of `permute/unsqueeze`.

static Tensor permute_inverse(const Tensor& self, IntArrayRef dims, InverseReturnMode inverse_return_mode) {
  // invert the permutation
  auto ndims = static_cast<int64_t>(dims.size());
  std::vector<int64_t> dims_(ndims);
  for(const auto i : c10::irange(ndims)) {
    dims_[at::maybe_wrap_dim(dims[i], ndims)] = i;
  }
  if (inverse_return_mode != InverseReturnMode::NeverView) {
    return at::permute(self, dims_);
  } else {
    return at::permute_copy(self, dims_);
  }
}

static Tensor unsqueeze_copy_to(const Tensor & self, c10::SymIntArrayRef sizes, InverseReturnMode inverse_return_mode) {
  auto result = self;
  bool need_alias = (inverse_return_mode == InverseReturnMode::AlwaysView);
  int64_t nDims = static_cast<int64_t>(sizes.size());
  for(const auto dim : c10::irange(nDims)) {
    if (sizes[dim] == 1) {
      need_alias = false;
      if (inverse_return_mode != InverseReturnMode::NeverView) {
        result = at::unsqueeze(result, dim);
      } else {
        result = at::unsqueeze_copy(result, dim);
      }
    }
  }

  // return an alias to ensure the output is a view when necessary
  return need_alias ? at::alias(result) : result;
}

static Tensor unsqueeze_copy_to(const Tensor & self, IntArrayRef dim, c10::SymIntArrayRef sizes, InverseReturnMode inverse_return_mode) {
  const auto ndim = static_cast<int64_t>(sizes.size());
  const auto mask = at::dim_list_to_bitset(dim, ndim);
  Tensor result = self;
  bool need_alias = (inverse_return_mode == InverseReturnMode::AlwaysView);
  // in NumPy it's not an error to unsqueeze a scalar, but we still need to avoided
  // unsqueezing in the backward.
  if (ndim == 0) {
    // return an alias to ensure the output is a view when necessary
    return need_alias ? at::alias(result) : result;
  }

  for (const auto d : c10::irange(ndim)) {
    if (mask.test(d) && sizes[d] == 1) {
      need_alias = false;
      if (inverse_return_mode != InverseReturnMode::NeverView) {
        result = at::unsqueeze(result, d);
      } else {
        result = at::unsqueeze_copy(result, d);
      }
    }
  }

  // return an alias to ensure the output is a view when necessary
  return need_alias ? at::alias(result) : result;
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

Tensor FunctionalInverses::_fw_primal_inverse(const at::Tensor& base, const at::Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t level) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call _fw_primal() during the functionalization pass. For now, this is not supported.");
}

Tensor FunctionalInverses::_make_dual_inverse(const at::Tensor& base, const at::Tensor& mutated_view, InverseReturnMode inverse_return_mode, const at::Tensor& tangent, int64_t level) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call _make_dual() during the functionalization pass. For now, this is not supported.");
}

Tensor FunctionalInverses::view_as_real_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      return at::view_as_complex(mutated_view);
    } else {
      return at::view_as_complex_copy(mutated_view);
    }
}

Tensor FunctionalInverses::view_as_complex_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      return at::view_as_real(mutated_view.resolve_conj());
    } else {
      return at::view_as_real_copy(mutated_view.resolve_conj());
    }
}

Tensor FunctionalInverses::_conj_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      return at::_conj(mutated_view);
    } else {
      return at::_conj_copy(mutated_view);
    }
}

Tensor FunctionalInverses::_neg_view_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      return at::_neg_view(mutated_view);
    } else {
      return at::_neg_view_copy(mutated_view);
    }
}

Tensor FunctionalInverses::as_strided_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, at::SymIntArrayRef size, at::SymIntArrayRef stride, std::optional<c10::SymInt> storage_offset) {
    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
      // NB: assumes mutated_view is a narrowed view of base.
      // We should NOT do this for functionalization
      return mutated_view.as_strided_symint(
          base.sym_sizes(), base.sym_strides(), base.sym_storage_offset());
    } else {
      return base.as_strided_scatter_symint(mutated_view, size, stride, std::move(storage_offset));
    }
}

Tensor FunctionalInverses::diagonal_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t offset, int64_t dim1, int64_t dim2) {
    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
      // NB: assumes mutated_view is a narrowed view of base.
      // We should NOT do this for functionalization
      return mutated_view.as_strided_symint(
          base.sym_sizes(), base.sym_strides(), base.sym_storage_offset());
    } else {
      return base.diagonal_scatter(mutated_view, offset, dim1, dim2);
    }
}

Tensor FunctionalInverses::expand_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, at::SymIntArrayRef size, bool implicit) {
    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
      // NB: assumes mutated_view is an expanded view of base.
      // We should NOT do this for functionalization
      return mutated_view.as_strided_symint(
          base.sym_sizes(), base.sym_strides(), base.sym_storage_offset());
    }
    // expand() can broadcast, so several elements of mutated_view may alias one element of base, and
    // the mutation may have written any subset of them. For each element of base, take the max over the
    // aliases that changed, or keep base's value if none did. In a well-defined program the aliases that
    // changed all hold the same value, so the max is exactly that value; if they don't, the largest wins.
    // (Summing the changes instead is wrong when several aliases were written, and inexact: inf - inf = nan.)
    // Limitations of detecting writes by comparing against base:
    // - A write of the value an alias already held looks like no write. The result is still correct, but
    //   if autograd differentiates this inverse, the gradient flows to base instead of the written value.
    //   torch.func.grad(functionalize(f)) does that; torch.compile doesn't, since AOTAutograd runs
    //   autograd above functionalization.
    // - Values are taken per element of base (per part for complex values). If a dtype view lets a single
    //   mutation write different bytes of one element through different aliases, only one alias's value
    //   is kept.
    // - amax doesn't preserve NaN payloads, so a NaN written through an alias may come back as a
    //   different NaN.
    const auto sizes = mutated_view.sym_sizes();
    const auto base_sizes = base.sym_sizes();
    const auto leading_dims = static_cast<int64_t>(sizes.size() - base_sizes.size());
    c10::SmallVector<int64_t, 8> broadcast_dims;
    for (int64_t d = 0; d < static_cast<int64_t>(sizes.size()); ++d) {
      if (TORCH_GUARD_OR_TRUE(sym_ne(sizes[d], 1)) &&
          (d < leading_dims || TORCH_GUARD_OR_FALSE(sym_eq(base_sizes[d - leading_dims], 1)))) {
        broadcast_dims.push_back(d);
      }
    }
    const bool never_view = inverse_return_mode == InverseReturnMode::NeverView;
    // Without broadcasting there are no aliases, and the exact inverse is a reshape.
    auto result = mutated_view;
    if (!broadcast_dims.empty()) {
      // The real and imaginary parts of an element can be written through different aliases.
      const bool is_complex = mutated_view.is_complex();
      auto as_real = [&](const Tensor& t) {
        return never_view ? at::view_as_real_copy(t.resolve_conj()) : at::view_as_real(t.resolve_conj());
      };
      auto mutated = is_complex ? as_real(mutated_view) : mutated_view;
      auto original = is_complex ? as_real(base) : base;
      // Compare floating point values by their bits, so that a NaN that stays the same is unchanged, and
      // so that -0.0 vs. 0.0 is a change.
      auto bits = [&](const Tensor& t) {
        if (!at::isFloatingType(t.scalar_type())) {
          return t;
        }
        const auto size = t.element_size();
        TORCH_INTERNAL_ASSERT(size == 1 || size == 2 || size == 4 || size == 8);
        const auto int_type = size == 1 ? kByte : size == 2 ? kShort : size == 4 ? kInt : kLong;
        // Reinterpreting sees the physical bits, so resolve the neg bit first.
        return never_view ? at::view_copy(t.resolve_neg(), int_type) : t.resolve_neg().view(int_type);
      };
      auto changed = bits(mutated).ne(bits(original));
      c10::Scalar lowest;
      AT_DISPATCH_V2(mutated.scalar_type(), "expand_inverse", AT_WRAP([&] {
        using limits = std::numeric_limits<scalar_t>;
        if constexpr (std::is_same_v<scalar_t, bool>) {
          lowest = false;
        } else if constexpr (limits::has_infinity) {
          lowest = -std::numeric_limits<double>::infinity();
        } else if constexpr (limits::is_integer) {
          lowest = static_cast<int64_t>(limits::lowest());
        } else {
          lowest = static_cast<double>(limits::lowest());
        }
      }), AT_EXPAND(AT_ALL_TYPES), kBool, kHalf, kBFloat16, AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
      auto masked = at::where(changed, mutated, lowest);
      for (const auto d : broadcast_dims) {
        // amax can't reduce an empty dim, so pad any dim that isn't known to be nonzero. (Don't ask the
        // shape env: under backed_size_oblivious a size may be assumed nonzero yet be 0 at runtime.)
        const auto static_size = sizes[d].maybe_as_int();
        if (!static_size.has_value() || *static_size == 0) {
          auto pad_sizes = masked.sym_sizes().vec();
          pad_sizes[d] = 1;
          masked = at::cat({masked, masked.new_full_symint(pad_sizes, lowest)}, d);
        }
      }
      result = at::where(
          changed.any(broadcast_dims, /*keepdim=*/true), masked.amax(broadcast_dims, /*keepdim=*/true), original);
      if (is_complex) {
        result = never_view ? at::view_as_complex_copy(result) : at::view_as_complex(result);
      }
    }
    result = never_view ? at::view_copy_symint(result, base_sizes) : result.contiguous().view_symint(base_sizes);
    // Keep base's layout: views replayed from the updated base may depend on it (e.g. view()). where()
    // takes its output layout from its first operand, which ones_like() gives base's layout. (Unlike
    // at::copy, this works for subclasses such as DTensor.)
    return base.is_contiguous_or_false() ? result : at::where(at::ones_like(base, at::kBool), result, base);
}

Tensor FunctionalInverses::permute_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, at::IntArrayRef dims) {
    return at::functionalization::permute_inverse(mutated_view, dims, inverse_return_mode);
}

Tensor FunctionalInverses::_reshape_alias_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, at::SymIntArrayRef size, at::SymIntArrayRef stride) {
    // Note that I'm directly calling reshape(), and ignoring the strides.
    // _reshape_alias() isn't available from user code, and is an implementation detail of reshape().
    // Specifically, passing in the strides directly can get us into trouble in cases like:
    // b = a[0]; c = b.reshape(...); c.add_(1); print(a)
    // When we eventually run the _reshape_alias_inverse() call here, if we were to pass in both sizes and strides,
    // The call would fail because `mutated_view` doesn't have enough bytes of storage.
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      return at::_reshape_alias_symint(mutated_view, base.sym_sizes(), base.sym_strides());
    } else {
      return at::_reshape_alias_copy_symint(mutated_view, base.sym_sizes(), base.sym_strides());
    }
}

Tensor FunctionalInverses::select_int_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t dim, c10::SymInt index) {
    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
      // NB: assumes mutated_view is a narrowed view of base.
      // We should NOT do this for functionalization
      return mutated_view.as_strided_symint(
          base.sym_sizes(), base.sym_strides(), base.sym_storage_offset());
    } else {
      return base.select_scatter_symint(mutated_view, dim, std::move(index));
    }
}

Tensor FunctionalInverses::detach_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    // the functionalization pass doesn't care about autograd metadata - as a view, I think detach() is just an identity function
    return mutated_view;
}

Tensor FunctionalInverses::lift_fresh_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    return mutated_view;
}

Tensor FunctionalInverses::slice_Tensor_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t dim, std::optional<c10::SymInt> start, std::optional<c10::SymInt> end, c10::SymInt step) {
    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
      // NB: assumes mutated_view is a narrowed view of base.
      // We should NOT do this for functionalization
      return mutated_view.slice_inverse_symint(
          base, dim, std::move(start), std::move(end), std::move(step));
    } else {
      return base.slice_scatter_symint(mutated_view, dim, std::move(start), std::move(end), std::move(step));
    }
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
Tensor FunctionalInverses::split_Tensor_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t mutated_view_idx, c10::SymInt split_size, int64_t dim) {
    // It would be nice if this logic could be reused from autograd's split_backward(), but I don't think it can.
    // For functionalization, we have only have one of the tensors from the TensorList outputted by split(), and we want to layer i
    // on top of the base tensor.
    // For autograd, we have all of the tensors outputted by split() and we just want to stack them.
    dim = at::maybe_wrap_dim(dim, base.dim());
    auto dim_size = base.sym_size(dim);
    auto start = split_size * mutated_view_idx;
    auto end = split_size + start;
    if (end > dim_size) end = std::move(dim_size);

    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
      // NB: assumes mutated_view is a narrowed view of base.
      // We should NOT do this for functionalization
      return mutated_view.slice_inverse_symint(base, dim, start, end, 1);
    } else {
      return base.slice_scatter_symint(mutated_view, dim, start, end, 1);
    }
}

Tensor FunctionalInverses::split_with_sizes_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t mutated_view_idx, c10::SymIntArrayRef split_sizes, int64_t dim) {
    dim = at::maybe_wrap_dim(dim, base.dim());
    auto dim_size = base.sym_size(dim);
    c10::SymInt start = 0;
    for (int64_t i = 0; i < mutated_view_idx; ++i) {
        start += split_sizes[i];
    }
    auto end = start + split_sizes[mutated_view_idx];
    if (end > dim_size) end = std::move(dim_size);

    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
      // NB: assumes mutated_view is a narrowed view of base.
      // We should NOT do this for functionalization
      return mutated_view.slice_inverse_symint(base, dim, start, end, 1);
    } else {
      return base.slice_scatter_symint(mutated_view, dim, start, end, 1);
    }
}

Tensor FunctionalInverses::squeeze_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    return unsqueeze_copy_to(mutated_view, base.sym_sizes(), inverse_return_mode);
}

Tensor FunctionalInverses::squeeze_dim_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t dim) {
    return unsqueeze_copy_to(mutated_view, dim, base.sym_sizes(), inverse_return_mode);
}

Tensor FunctionalInverses::squeeze_dims_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, IntArrayRef dim) {
    return unsqueeze_copy_to(mutated_view, dim, base.sym_sizes(), inverse_return_mode);
}

Tensor FunctionalInverses::t_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      return at::t(mutated_view);
    } else {
      return at::t_copy(mutated_view);
    }
}

Tensor FunctionalInverses::transpose_int_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t dim0, int64_t dim1) {
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      return transpose(mutated_view, dim0, dim1);
    } else {
      return transpose_copy(mutated_view, dim0, dim1);
    }
}

Tensor FunctionalInverses::_nested_view_from_buffer_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, const Tensor& nested_sizes, const Tensor& nested_strides, const Tensor& storage_offsets) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call _nested_view_from_buffer() during the functionalization pass. For now, nested tensors aren't supported during functionalization");
}

Tensor FunctionalInverses::_nested_view_from_jagged_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, const Tensor& offsets, const Tensor& dummy, const std::optional<Tensor>& lengths, int64_t ragged_idx, const std::optional<Tensor>& min_seqlen, const std::optional<Tensor>& max_seqlen) {
  auto values = at::_nested_get_values(mutated_view);
  if (inverse_return_mode != InverseReturnMode::NeverView) {
    return values;
  } else {
    return values.clone(/*memory_format=*/at::MemoryFormat::Contiguous);
  }
}

Tensor FunctionalInverses::_nested_get_values_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
  auto offsets = at::_nested_get_offsets(base);
  auto lengths = at::_nested_get_lengths(base);
  auto ragged_idx = at::_nested_get_ragged_idx(base);
  auto dummy = at::_nested_get_jagged_dummy(base);
  auto min_seqlen = at::_nested_get_min_seqlen(base);
  auto max_seqlen = at::_nested_get_max_seqlen(base);
  auto nt = at::_nested_view_from_jagged(
      mutated_view, offsets, dummy, lengths, ragged_idx,
      (min_seqlen.defined() ? std::optional<Tensor>(min_seqlen) : std::nullopt),
      (max_seqlen.defined() ? std::optional<Tensor>(max_seqlen) : std::nullopt));

  if (inverse_return_mode != InverseReturnMode::NeverView) {
    return nt;
  } else {
    return nt.clone(/*memory_format=*/at::MemoryFormat::Contiguous);
  }
}

Tensor FunctionalInverses::unsqueeze_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t dim) {
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      return at::squeeze(mutated_view, dim);
    } else {
      return at::squeeze_copy(mutated_view, dim);
    }
}

Tensor FunctionalInverses::_indices_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call _indices() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
}

Tensor FunctionalInverses::_values_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call _values() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
}

Tensor FunctionalInverses::indices_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call indices() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
}

Tensor FunctionalInverses::values_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call values() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
}

Tensor FunctionalInverses::_sparse_broadcast_to_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, at::IntArrayRef size) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call _sparse_broadcast_to() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
}

Tensor FunctionalInverses::crow_indices_inverse(const at::Tensor& base, const at::Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call crow_indices() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
}

Tensor FunctionalInverses::col_indices_inverse(const at::Tensor& base, const at::Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call col_indices() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
}

Tensor FunctionalInverses::ccol_indices_inverse(const at::Tensor& base, const at::Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call ccol_indices() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
}

Tensor FunctionalInverses::row_indices_inverse(const at::Tensor& base, const at::Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    TORCH_INTERNAL_ASSERT(false, "Attempted to call row_indices() during the functionalization pass. For now, sparse tensors aren't supported during functionalization");
}

Tensor FunctionalInverses::unbind_int_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t mutated_view_idx, int64_t dim) {
    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
      // NB: assumes mutated_view is a narrowed view of base.
      // We should NOT do this for functionalization
      return mutated_view.as_strided_symint(
          base.sym_sizes(), base.sym_strides(), base.sym_storage_offset());
    } else {
      dim = at::maybe_wrap_dim(dim, static_cast<int64_t>(base.sizes().size()));
      return base.select_scatter(mutated_view, dim, mutated_view_idx);
    }
}

Tensor FunctionalInverses::view_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, at::SymIntArrayRef size) {
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      return mutated_view.view_symint(base.sym_sizes());
    } else {
      return at::view_copy_symint(mutated_view, base.sym_sizes());
    }
}


Tensor FunctionalInverses::view_dtype_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, at::ScalarType dtype) {
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      return mutated_view.view(base.scalar_type());
    } else {
      return at::view_copy(mutated_view, base.scalar_type());
    }
}

Tensor FunctionalInverses::unfold_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode, int64_t dimension, int64_t size, int64_t step) {
    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
      // NB: assumes mutated_view is a narrowed view of base.
      // We should NOT do this for functionalization
      return mutated_view.as_strided_symint(
          base.sym_sizes(), base.sym_strides(), base.sym_storage_offset());
    } else {
      // I think autograd and the functionalization pass want the exact same thing here, but need to test to confirm.
      // unfold_backward() is safe to use here because it is NOT a view op.
      // (note: technically, we'll have an extra memory copy.
      // We'd need to add an aliasing version of unfold_backward to fix that though).
      TORCH_CHECK(
        !(inverse_return_mode == InverseReturnMode::ViewOrScatterInverse && size > step),
        "While executing unfold, functionalization encountered a tensor being mutated that has internal overlap. \
When using torch.compile (or running functionalization directly), this is banned \
as the behavior is not well defined. Consider cloning the tensor before mutating it, \
or removing the mutation from your model."
          );
      return unfold_backward(mutated_view, base.sizes(), dimension, size, step);
    }
}

Tensor FunctionalInverses::alias_inverse(const Tensor& base, const Tensor& mutated_view, InverseReturnMode inverse_return_mode) {
    if (inverse_return_mode != InverseReturnMode::NeverView) {
      return at::alias(mutated_view);
    } else {
      return at::alias_copy(mutated_view);
    }
}

Tensor FunctionalInverses::chunk_inverse(const at::Tensor & base, const at::Tensor & mutated_view, InverseReturnMode inverse_return_mode, int64_t mutated_view_idx, int chunks, int dim) {
    // TODO: Can the logic from TensorShape.cpp be reused here somehow?
    const auto dim_size = base.sym_size(dim);
    auto split_size = (dim_size + chunks - 1) / chunks;
    std::vector<c10::SymInt> split_sizes(chunks, split_size);
    split_sizes[chunks - 1] = split_size - (split_size * chunks - dim_size);
    return split_with_sizes_inverse(base, mutated_view, inverse_return_mode, mutated_view_idx, split_sizes, dim);
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
Tensor FunctionalInverses::narrow_inverse(const at::Tensor & base, const at::Tensor & mutated_view, InverseReturnMode inverse_return_mode, int dim, c10::SymInt start, c10::SymInt length) {
    if (inverse_return_mode == InverseReturnMode::AlwaysView) {
      // NB: assumes mutated_view is a narrowed view of base.
      // We should NOT do this for functionalization
      return mutated_view.slice_inverse_symint(base, dim, start, start + length, 1);
    } else {
      return base.slice_scatter_symint(
          mutated_view, dim, start, start + length, 1);
    }
}

Tensor FunctionalInverses::slice_inverse_inverse(const at::Tensor & base, const at::Tensor & mutated_view, InverseReturnMode inverse_return_mode, const at::Tensor & src, int64_t dim, std::optional<c10::SymInt> start, std::optional<c10::SymInt> end, c10::SymInt step) {
    // slice_inverse() inverse is just slice()
    if (inverse_return_mode == InverseReturnMode::NeverView) {
      return at::slice_copy_symint(
          mutated_view, dim, std::move(start), std::move(end), std::move(step));
    } else {
      return mutated_view.slice_symint(
          dim, std::move(start), std::move(end), std::move(step));
    }
}

} // namespace at::functionalization
