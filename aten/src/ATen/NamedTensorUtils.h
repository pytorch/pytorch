#pragma once
#ifdef BUILD_NAMEDTENSOR

#include <ATen/NamedTensor.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/DimVector.h>
#include <functional>

namespace at {

using NameVector = SmallVector<Dimname, kDimVectorStaticSize>;

inline bool has_names(TensorList tensors) {
  return std::any_of(
      tensors.begin(), tensors.end(), [](const Tensor& t) { return t.has_names(); });
}

// Converts dim to an positional index. Errors if `dim` cannot be used to
// refer to any dimension of tensor.
CAFFE2_API int64_t dimname_to_position(const Tensor& tensor, Dimname dim);
CAFFE2_API std::vector<int64_t> dimnames_to_positions(const Tensor& tensor, DimnameList dims);

// Unifies two DimnameList to produce a third. This is useful for implementing
// the named inference rule for binary broadcasting operations like add.
//
// There are three main constraints:
// 1) Check matching: Names must match positionally from the right.
// 2) Check misaligned: If a name `n` is in `names`, then it must appear at
//    the same index from the right in other.
// 3) The output names are obtained by unifying the names individually from the right.
CAFFE2_API optional<std::vector<Dimname>>
unify_from_right(optional<DimnameList> names, optional<DimnameList> other);

namespace namedinference {

// Names get propagated via the following rules:
// 1) If result does not have names, then `names` get propagated.
// 2) If result has names, then `names` must be equal to result.names
void propagate_names(Tensor& result, optional<DimnameList> names);
void propagate_names(Tensor& result, std::vector<Dimname>&& names, bool validate_names);
void propagate_names(TensorImpl* result, optional<DimnameList> names);
void propagate_names(TensorImpl* result, std::vector<Dimname>&& names, bool validate_names);

// Propagates all names from src to result.
void propagate_names(Tensor& result, const Tensor& src);
void propagate_names(TensorImpl* result, /*const */TensorImpl* src);

// Propagates all names except for those at the excluded_idxs.
void propagate_names_except(Tensor& result, const Tensor& src, IntArrayRef excluded_idxs);

// Used for reduction ops that have a `keepdim` arg.
void propagate_names_for_reduction(Tensor& result, const Tensor& src, IntArrayRef excluded_idxs, bool keepdim);

} // namespace namedinference

} // namespace at
#endif
