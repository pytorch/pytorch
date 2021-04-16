#pragma once

#include <ATen/WrapDimUtils.h>

namespace at { namespace namedinference {


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
// Definition: unify(name, other) fails if the names do not match. Otherwise,
// it returns the most refined of name and other.
//
// Here is an example of checking if two names match.
// tensor: Tensor[A, None]
// other: Tensor[A]
//
// Let's say we wish to check if tensor.names[-1] matches other.names[-1].
// None (in tensor) cannot match A (in other) because if the None were refined
// to A, `tensor` would have duplicate names [A, A]. Therefore we need to check
// tensor.names [A, None] for the existence of A.
struct TORCH_API TensorName {
  explicit TensorName(ArrayRef<Dimname> origin, int origin_idx)
    : origin_(origin),
      name_(origin[maybe_wrap_dim(origin_idx, origin.size())]),
      origin_idx_(origin_idx) {}

  // op_name is only used for error reporting.
  const TensorName& unify(const TensorName& other, const char* op_name) const;
  Dimname toDimname() const;

 private:
  ArrayRef<Dimname> origin_;
  Dimname name_;
  int origin_idx_; // A named tensor can have at most 64 dims.

  TORCH_API friend std::ostream& operator<<(
      std::ostream& out,
      const TensorName& tensorname);
};

using TensorNameVec = SmallVector<TensorName, 10>;

struct TORCH_API TensorNames {
  explicit TensorNames(ArrayRef<Dimname> names);

  // Create TensorNames from names[start:end]. Each individual TensorName stores
  // `names`, NOT names[start:end], because the original tensor's names are `names`.
  explicit TensorNames(ArrayRef<Dimname> names, int64_t start, int64_t end);

  // op_name is only used for error reporting.
  TensorNames& unifyFromRightInplace(
      const TensorNames& other,
      const char* op_name = "unify");
  void checkUnique(const char* op_name) const;

  void append(TensorName&& name);
  std::vector<Dimname> toDimnameVec() const;

 private:
  explicit TensorNames(TensorNameVec&& names) : names_(names) {};

  TensorNameVec names_;
};


}} // namespace at::namedinference
