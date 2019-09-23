#pragma once
#include <ATen/core/EnableNamedTensor.h>
#include <ATen/NamedTensor.h>

#include <ATen/core/Tensor.h>
#include <ATen/core/DimVector.h>
#include <functional>

#ifdef BUILD_NAMEDTENSOR
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
CAFFE2_API std::vector<Dimname>
unify_from_right(DimnameList names, DimnameList other, const char* action = "broadcast");

namespace namedinference {

// Names get propagated via the following rules:
// 1) If result does not have names, then `names` get propagated.
// 2) If result has names, then `names` must be equal to result.names
void propagate_names(Tensor& result, optional<DimnameList> names);
void propagate_names(Tensor& result, std::vector<Dimname>&& names, bool validate_names);
CAFFE2_API void propagate_names(Tensor& result, optional<std::vector<Dimname>>&& maybe_names, bool validate_names);
void propagate_names(TensorImpl* result, optional<DimnameList> names);
void propagate_names(TensorImpl* result, std::vector<Dimname>&& names, bool validate_names);
void propagate_names(TensorImpl* result, optional<std::vector<Dimname>>&& maybe_names, bool validate_names);

// Propagates all names from src to result.
void propagate_names(Tensor& result, const Tensor& src);
void propagate_names(TensorImpl* result, /*const */TensorImpl* src);

// Propagates all names except for those at the excluded_idxs.
void propagate_names_except(Tensor& result, const Tensor& src, IntArrayRef excluded_idxs);

// Used for reduction ops that have a `keepdim` arg.
void propagate_names_for_reduction(Tensor& result, const Tensor& src, IntArrayRef excluded_idxs, bool keepdim);

// Tensor::copy_ name inference rule.
void propagate_names_for_copy(Tensor& result, const Tensor& src);

// result = m1 @ m2 + bias
void propagate_names_for_addmm(
    TensorImpl* result,
    /*const*/TensorImpl* m1,
    /*const*/TensorImpl* m2,
    /*const*/TensorImpl* bias);

void propagate_names_for_addmv(
    TensorImpl* result,
    TensorImpl* mat,
    TensorImpl* vec,
    TensorImpl* bias);

void check_names_for_dot(TensorImpl* vec1, TensorImpl* vec2);

void propagate_names_for_expand(Tensor& result, const Tensor& self);

optional<std::vector<Dimname>> compute_cat_outnames(TensorList tensors);

optional<std::vector<Dimname>> compute_broadcast_outnames(
    const Tensor& self,
    const Tensor& other);

optional<std::vector<Dimname>> broadcast_to_outnames(
    const Tensor& tensor,
    const Tensor& reference_tensor,
    const char* op_name);

optional<std::vector<Dimname>> compute_baddbmm_outnames(
    TensorImpl* result,
    TensorImpl* self,
    TensorImpl* other,
    TensorImpl* bias);

optional<std::vector<Dimname>> compute_matmul_outnames(const Tensor& self, const Tensor& other);

optional<std::vector<Dimname>> compute_bmm_outnames(
    Tensor& result,
    const Tensor& self,
    const Tensor& other);

} // namespace namedinference

} // namespace at
#endif
