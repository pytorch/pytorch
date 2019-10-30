#pragma once

#include <ATen/core/EnableNamedTensor.h>
#include <ATen/WrapDimUtils.h>

namespace at { namespace namedinference {

#ifdef BUILD_NAMEDTENSOR

// TensorName and TensorNames are wrappers around Dimname and DimnameList
// that contain helper functions to make writing name inference rules easier.
//
// A TensorName represents a Dimname associated with some DimnameList (from a Tensor).
// This encapsulates all the information that is needed to check if names *match*
// and to *unify* names.
//
// Definition: Two names in two tensors *match* if they are equal, or if at
// least one of them is a wildcard that can be *refined* to the other name.
//
// Here is an example of checking if two names match.
// tensor: Tensor[A, None]
// other: Tensor[A]
//
// Let's say we wish to check if tensor.names[-1] matches other.names[-1].
// None (in tensor) cannot match A (in other) because if the None were refined
// to A, `tensor` would have duplicate names [A, A]. Therefore we need to check
// tensor.names [A, None] for the existence of A.
struct CAFFE2_API TensorName {
  explicit TensorName(ArrayRef<Dimname> origin, int origin_idx)
    : origin_(origin),
      origin_idx_(origin_idx),
      name_(origin[maybe_wrap_dim(origin_idx, origin.size())]) {}

  const TensorName& unify(const TensorName& other, const char* op_name) const;
  Dimname toDimname() const;

 private:
  ArrayRef<Dimname> origin_;
  int origin_idx_;
  Dimname name_;
};

using TensorNameVec = SmallVector<TensorName, 10>;

struct CAFFE2_API TensorNames {
  explicit TensorNames(ArrayRef<Dimname> names);

  // Create TensorNames from names[start:end]. Each individual TensorName stores
  // `names`, NOT names[start:end], because the original tensor's names are `names`.
  explicit TensorNames(ArrayRef<Dimname> names, int64_t start, int64_t end);

  TensorNames unifyFromRight(const TensorNames& other, const char* op_name) const;

  void append(TensorName&& name);
  std::vector<Dimname> toDimnameVec() const;

 private:
  explicit TensorNames(TensorNameVec&& names) : names_(names) {};

  TensorNameVec names_;
};
#endif

}} // namespace at::namedinference
