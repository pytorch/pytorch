#pragma once

#include <ATen/core/Dimname.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/C++17.h>

namespace at {

// XXX: This file exists because TensorImpl is in c10, but Dimname is in ATen.
// Due to the c10/ATen library split, TensorImpl cannot depend on Dimname,
// so we have a couple of workarounds.
//
// In the long term, we'll move Dimname to c10 and everything in this file
// can be refactored out. The main blocker for that is that "c10::Symbol"
// actually exists outside of c10 and needs to be moved in.

// TensorImpl has a unique_ptr<NamedTensorMetaInterface> field.
// XXX: Ideally we would just put optional<vector<Dimname>> into TensorImpl.
//
// This class has an important invariant: there must be at least ONE
// non-wildcard
struct TORCH_API NamedTensorMeta final : public c10::NamedTensorMetaInterface {
  // This enum is to remind people that the invariant on constructors is that
  // the list of dimnames must have at least one non-wildcard
  enum HAS_NON_WILDCARD {
    HasNonWildcard
  };

  explicit NamedTensorMeta(HAS_NON_WILDCARD, DimnameList names)
    : names_(names.vec()) {
    check_invariants();
  }
  explicit NamedTensorMeta(HAS_NON_WILDCARD, std::vector<Dimname>&& names)
    : names_(std::move(names)) {
    check_invariants();
  }

  std::unique_ptr<c10::NamedTensorMetaInterface> clone() const override {
    return std::make_unique<NamedTensorMeta>(HasNonWildcard, names_);
  }

  DimnameList names() const { return names_; }

  // Used for an assertion in TensorImpl.h
  int64_t slow_dim() const override {
    return names_.size();
  }

  void check_invariants() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      std::any_of(names_.begin(), names_.end(), [](const Dimname& n) { return !n.isWildcard(); }));
  }

  void set_names(HAS_NON_WILDCARD, DimnameList new_names) {
    TORCH_INTERNAL_ASSERT(new_names.size() == names_.size());
    std::copy(new_names.begin(), new_names.end(), names_.begin());
    check_invariants();
  }

  void set_names(HAS_NON_WILDCARD, std::vector<Dimname>&& new_names) {
    TORCH_INTERNAL_ASSERT(new_names.size() == names_.size());
    names_ = std::move(new_names);
    check_invariants();
  }

  // INVARIANT: at least one Dimname is non-WILDCARD
  std::vector<Dimname> names_;
};

// When NamesMode is disabled, then all operations ignore tensors' names fields.
// Concretely speaking, all tensors are treated as having nullopt names.
struct TORCH_API NamesMode {
  static bool is_enabled();
  static void set_enabled(bool enabled);
};


// A RAII, thread local (!) guard that enables or disables names upon
// construction, and sets it back to the original value upon destruction.
struct TORCH_API NoNamesGuard {
  NoNamesGuard() : prev_mode(NamesMode::is_enabled()), initialized(true) {
    NamesMode::set_enabled(false);
  }
  ~NoNamesGuard() {
    if (initialized) {
      reset();
    }
  }
  void reset() {
    TORCH_INTERNAL_ASSERT(initialized);
    NamesMode::set_enabled(prev_mode);
  }
 private:
  bool prev_mode;
  bool initialized;
};

void check_names_valid_for(const Tensor& tensor, DimnameList names);
void check_names_valid_for(size_t tensor_dim, DimnameList names);

// Sets the names of `tensor` to be `names`.
TORCH_API Tensor& internal_set_names_inplace(Tensor& tensor, c10::optional<DimnameList> names);
TORCH_API Tensor& internal_set_names_inplace(Tensor& tensor, std::vector<Dimname>&& names, bool validate_names);

constexpr size_t kMaxNamedTensorDim = 64;

DimnameList default_names(size_t len);

namespace impl {

// Some helper functions on TensorImpl. Useful for working with names in TH.
// XXX: Ideally these would exist as methods on TensorImpl
TORCH_API void internal_set_names_inplace(TensorImpl* impl, c10::optional<DimnameList> names, bool validate_names);
TORCH_API void internal_set_names_inplace(TensorImpl* impl, std::vector<Dimname>&& names, bool validate_names);

void check_names_valid_for(TensorImpl* impl, DimnameList names);

// Returns true if the tensor's names exist and are not all 'None'.
// Returns false if the tensor's names don't exist (were not allocated),
// or if all names are 'None'.
// We treat not-allocated-names the same as allocated names that are all 'None'.
TORCH_API bool has_names(const TensorImpl* impl);

// Returns the names of the tensor's dimensions.
// Unnamed tensors are treated as having 'None' in all dimension; this method
// would return a DimnameList of all 'None's for an unnamed tensor.
TORCH_API DimnameList get_names(const TensorImpl* impl);

// This is more of an implementation detail; one should use impl::get_names /
// Tensor::names() whenever possible because it provides a cleaner API.
// Returns the names of the tensor if they have been allocated; returns nullopt
// instead if the haven't been. The names of a tensor are not allocated if a
// tensor is constructed with names=None.
TORCH_API c10::optional<DimnameList> get_opt_names(const TensorImpl* impl);


} // namespace impl

} // namespace at
