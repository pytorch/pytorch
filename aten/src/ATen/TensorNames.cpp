#include <ATen/core/EnableNamedTensor.h>
#include <ATen/TensorNames.h>
#include <ATen/WrapDimUtils.h>

namespace at { namespace namedinference {

#ifdef BUILD_NAMEDTENSOR

Dimname TensorName::toDimname() const {
  return name_;
}

const TensorName& TensorName::unify(const TensorName& other, const char* op_name) const {
  // unify(None, None)
  if (name_.isWildcard() && other.name_.isWildcard()) {
    return *this;
  }

  // unify(A, A)
  if (name_ == other.name_) {
    return *this;
  }

  // unify(A, None)
  if (other.name_.isWildcard()) {
    const auto it = std::find(other.origin_.begin(), other.origin_.end(), name_);
    TORCH_CHECK(it == other.origin_.end(),
        op_name, ":",
        " Cannot match ", name_,
        " (at index ", origin_idx_, " of ", origin_, ")",
        " with ", other.name_,
        " (at index ", other.origin_idx_, " of ", other.origin_, ")",
        " because the latter names already has ", name_, ".",
        " Are your tensors misaligned?");
    return *this;
  }

  // unify(None, A)
  if (name_.isWildcard()) {
    return other.unify(*this, op_name);
  }

  // unify(A, B)
  TORCH_CHECK(name_ == other.name_,
      op_name, ":",
      " Expected ", name_,
      " (at index ", origin_idx_, " of ", origin_, ")",
      " to match ", other.name_,
      " (at index ", other.origin_idx_, " of ", other.origin_, ")",
      " but they do not match.");
  return *this;
}

TensorNames::TensorNames(ArrayRef<Dimname> names) {
  names_.reserve(names.size());
  for (int64_t idx = 0; idx < names.size(); ++idx) {
    names_.emplace_back(names, idx);
  }
}

TensorNames::TensorNames(ArrayRef<Dimname> names, int64_t start, int64_t end) {
  start = maybe_wrap_dim(start, names.size());
  end = maybe_wrap_dim(end, names.size());
  names_.reserve(end - start);
  for (int64_t idx = start; idx < end; ++idx) {
    names_.emplace_back(names, idx);
  }
}

TensorNames TensorNames::unifyFromRight(const TensorNames& other, const char* op_name) const {
  const auto longer_size = std::max(names_.size(), other.names_.size());
  TensorNameVec result;
  result.reserve(longer_size);

  const auto& longer = names_.size() == longer_size ? names_ : other.names_;
  const auto& shorter = names_.size() == longer_size ? other.names_ : names_;
  const auto size_difference = longer_size - shorter.size();

  result.insert(result.begin(), longer.begin(), longer.begin() + size_difference);

  for (int64_t idx = size_difference; idx < longer_size; ++idx) {
    result.push_back(longer[idx].unify(shorter[idx - size_difference], op_name));
  }

  return TensorNames(std::move(result));
}

void TensorNames::append(TensorName&& name) {
  names_.emplace_back(name);
}

std::vector<Dimname> TensorNames::toDimnameVec() const {
  std::vector<Dimname> result;
  result.reserve(names_.size());
  for (const auto& tensor_name : names_) {
    result.emplace_back(tensor_name.toDimname());
  }
  return result;
}

#endif

}} // namespace at::namedinference
