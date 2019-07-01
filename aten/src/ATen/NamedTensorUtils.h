#pragma once
#ifdef BUILD_NAMEDTENSOR

#include <ATen/NamedTensor.h>
#include <ATen/core/Tensor.h>
#include <functional>

namespace at {

inline bool has_names(TensorList tensors) {
  return std::any_of(
      tensors.begin(), tensors.end(), [](const Tensor& t) { return t.is_named(); });
}

// Sets the names of `tensor` to be `names`.
CAFFE2_API void internal_set_names_inplace(Tensor& tensor, optional<DimnameList> names);

// Converts dim to an positional index. Errors if `dim` cannot be used to
// refer to any dimension of tensor.
CAFFE2_API int64_t dimname_to_position(const Tensor& tensor, Dimname dim);

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

optional<std::vector<Dimname>> erase_name(optional<DimnameList> self_names, int64_t dim);
void propagate_names(Tensor& result, const Tensor& src);
void propagate_names(TensorImpl* result, /*const */TensorImpl* src);

} // namespace namedinference

} // namespace at
#endif
