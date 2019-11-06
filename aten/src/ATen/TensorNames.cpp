#include <ATen/core/EnableNamedTensor.h>
#include <ATen/TensorNames.h>
#include <ATen/WrapDimUtils.h>

namespace at { namespace namedinference {

#ifdef BUILD_NAMEDTENSOR

Dimname TensorName::toDimname() const {
  TORCH_INTERNAL_ASSERT(initialized_);
  return name_;
}

const TensorName& TensorName::unify(const TensorName& other, const char* op_name) const {
  TORCH_INTERNAL_ASSERT(initialized_);

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
        " Cannot match ", *this, " with ", other,
        " because the latter names already have ", name_, ".",
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
      " Expected ", *this,
      " to match ", other,
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

TensorNames& TensorNames::unifyFromRightInplace(const TensorNames& other, const char* op_name) {
  int64_t longer_size = std::max(names_.size(), other.names_.size());
  bool this_is_longer = names_.size() == longer_size;

  ArrayRef<TensorName> longer = this_is_longer ? names_ : other.names_;
  ArrayRef<TensorName> shorter = this_is_longer ? other.names_ : names_;
  int64_t size_difference = longer_size - shorter.size();

  names_.resize(longer_size);

  // perform unification, starting on the right, on (longer_size - size_difference)
  // number of elements.
  for (int64_t idx = longer_size - 1; idx >= size_difference; --idx) {
    names_[idx] = longer[idx].unify(shorter[idx - size_difference], op_name);
  }

  // Copy over the remaining elements.
  if (!this_is_longer) {
    for (int64_t idx = 0; idx < size_difference; ++idx) {
      names_[idx] = longer[idx];
    }
  }

  return *this;
}

void TensorNames::append(TensorName&& name) {
  names_.emplace_back(name);
}

void TensorNames::checkUnique(const char* op_name) const {
  // O(N^2), but named tensors can have at most N = 64 dimensions, so this
  // doesn't matter unless benchmarking tells us it does. The alternative is
  // to create some sort of set data structure but the overhead of that
  // might dominate for small sizes.
  for (auto it = names_.begin(); it != names_.end(); ++it) {
    const auto name = it->toDimname();
    if (name.isWildcard()) continue;

    auto dup = std::find_if(it + 1, names_.end(),
        [&](const TensorName& other) { return other.toDimname() == name; });
    TORCH_CHECK(dup == names_.end(),
        op_name, ": ",
        "Attempted to propagate dims ", *it, " and ", dup, " to the output, ",
        "but that would create a tensor with duplicate names ", toDimnameVec(),
        ". Please rename your inputs with Tensor.rename to prevent this.");
  }
}

// Let's say the TensorName represents 'C' in ['N', 'C', 'H, 'W'].
// It should print like:
// 'C' (index 1 of ['N', 'C', 'H', 'W'])
std::ostream& operator<<(std::ostream& out, const TensorName& tensorname) {
  out << tensorname.name_ << " (index ";
  out << static_cast<int>(tensorname.origin_idx_) << " of ";
  out << tensorname.origin_ << ")";
  return out;
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
