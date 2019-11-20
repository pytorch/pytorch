#pragma once
#include <ATen/core/EnableNamedTensor.h>
#include <ATen/NamedTensor.h>
#include <ATen/TensorNames.h>

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

[[noreturn]] inline void reportNYIDimnameOverload(const char* op_name) {
  TORCH_CHECK(
      false,
      op_name, ": You passed a dimname (string) to this op in place of a dimension "
      "index but it does not yet support this behavior. Please pass a dimension "
      "index to work around this.");
}

// [NOTE] Writing name inference rules
//
// Operators that support named tensors are either composed of operations that
// support named tensors or implement some name inference rule. An op that
// implements its own name inference rule generally looks like the following:
//
// Tensor op(...) {
//   perform_shape_checks(...);
//   # (1)
//   auto maybe_outnames = compute_outnames(...);
//   auto result = [&]() {
//     NoNamesGuard guard;
//     return op_impl(...);
//   }();
//   # (2)
//   propagate_names_if_nonempty(result, maybe_outnames);
//
// Each op has (1) a compute outnames step and (2) a propagate names step.
//
// compute_outnames is responsible for checking that input names match and
// determining what the output names should be. It returns either:
// - {} (if the inputs tensors are all unnamed)
// - non-empty outnames.
//
// propagate_names_if_nonempty propagates the outnames if they exist to the result
// tensors.
//
// The {} case is an optimization; if the user does not use named tensors they
// pay no perf cost for it.

namespace namedinference {

// Propagates `names` to `result` if `names` is not empty.
// `names` can be empty; see [NOTE] Writing name inference rules
// If `names` is not empty, `names.size()` should equal `result.dim()`.
// When in doubt, use this overload instead of the others.
CAFFE2_API Tensor& propagate_names_if_nonempty(
    Tensor& result,
    DimnameList maybe_names,
    bool validate_names = false);

// Propagates `names` to `result`. Only use this if we are certain that there are
// names to propagate (that names is not empty).
CAFFE2_API Tensor& propagate_names(
    Tensor& result,
    DimnameList names,
    bool validate_names = false);

// Propagates all names from src to result.
CAFFE2_API void propagate_names(Tensor& result, const Tensor& src);

// Propagates all names except for those at the excluded_idxs.
void propagate_names_except(Tensor& result, const Tensor& src, IntArrayRef excluded_idxs);

// Used for reduction ops that have a `keepdim` arg.
void propagate_names_for_reduction(Tensor& result, const Tensor& src, IntArrayRef excluded_idxs, bool keepdim);

void propagate_names_for_expand(Tensor& result, const Tensor& self);

std::vector<Dimname> compute_cat_outnames(TensorList tensors);

std::vector<Dimname> compute_broadcast_outnames(
    const Tensor& self,
    const Tensor& other);

std::vector<Dimname> broadcast_to_outnames(
    const Tensor& tensor,
    const Tensor& reference_tensor,
    const char* op_name);

std::vector<Dimname> compute_matmul_outnames(const Tensor& self, const Tensor& other);

std::vector<Dimname> compute_cdist_outnames(const Tensor& self, const Tensor& other);

std::vector<Dimname> compute_bmm_outnames(
    Tensor& result,
    const Tensor& self,
    const Tensor& other);

std::vector<Dimname> compute_squeeze_outnames(const Tensor& tensor);

// TensorImpl* overloads for Legacy TH/THC code. Use these sparingly.

TensorImpl* propagate_names_if_nonempty(
    TensorImpl* result,
    DimnameList maybe_names,
    bool validate_names = false);

TensorImpl* propagate_names(
    TensorImpl* result,
    DimnameList names,
    bool validate_names = false);

void propagate_names(TensorImpl* result, /*const */TensorImpl* src);

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

std::vector<Dimname> compute_baddbmm_outnames(
    TensorImpl* result,
    TensorImpl* self,
    TensorImpl* other,
    TensorImpl* bias);

bool are_names_equal(TensorImpl* self, TensorImpl* other);

} // namespace namedinference

} // namespace at
#endif
