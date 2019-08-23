#pragma once
#ifdef BUILD_NAMEDTENSOR

#include <ATen/Dimname.h>
#include <c10/core/TensorImpl.h>
#include <torch/csrc/utils/memory.h>

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
struct CAFFE2_API NamedTensorMeta : public c10::NamedTensorMetaInterface {
  explicit NamedTensorMeta(int64_t num_names)
    : names_(std::vector<Dimname>(num_names, Dimname::wildcard())) {}

  explicit NamedTensorMeta(DimnameList names)
    : names_(names.vec()) {}
  explicit NamedTensorMeta(std::vector<Dimname>&& names)
    : names_(std::move(names)) {}

  std::unique_ptr<c10::NamedTensorMetaInterface> clone() const override {
    return torch::make_unique<NamedTensorMeta>(names_);
  }

  bool has_names() const;
  DimnameList names() const { return names_; }

  void set_names(DimnameList new_names) {
    TORCH_INTERNAL_ASSERT(new_names.size() == names_.size());
    std::copy(new_names.begin(), new_names.end(), names_.begin());
  }

  void set_names(std::vector<Dimname>&& new_names) {
    TORCH_INTERNAL_ASSERT(new_names.size() == names_.size());
    names_ = std::move(new_names);
  }

 private:
  std::vector<Dimname> names_;
};

// When NamesMode is disabled, then all operations ignore tensors' names fields.
// Concretely speaking, all tensors are treated as having nullopt names.
struct CAFFE2_API NamesMode {
  static bool is_enabled();
  static void set_enabled(bool enabled);
};

// A RAII, thread local (!) guard that enables or disables names upon
// construction, and sets it back to the original value upon destruction.
struct CAFFE2_API NoNamesGuard {
  NoNamesGuard() : prev_mode(NamesMode::is_enabled()) {
    NamesMode::set_enabled(false);
  }
  ~NoNamesGuard() {
    NamesMode::set_enabled(prev_mode);
  }
 private:
  bool prev_mode;
};


// Sets the names of `tensor` to be `names`.
CAFFE2_API Tensor& internal_set_names_inplace(Tensor& tensor, optional<DimnameList> names);
CAFFE2_API Tensor& internal_set_names_inplace(Tensor& tensor, std::vector<Dimname>&& names, bool validate_names);

constexpr size_t kMaxNamedTensorDim = 64;

DimnameList default_names(size_t len);

namespace impl {

// Some helper functions on TensorImpl. Useful for working with names in TH.
// XXX: Ideally these would exist as methods on TensorImpl
CAFFE2_API void internal_set_names_inplace(TensorImpl* impl, optional<DimnameList> names);
CAFFE2_API void internal_set_names_inplace(TensorImpl* impl, std::vector<Dimname>&& names, bool validate_names);
CAFFE2_API optional<DimnameList> get_opt_names(TensorImpl* impl);
CAFFE2_API bool has_names(TensorImpl* impl);


} // namespace impl

} // namespace at
#endif
