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

  std::unique_ptr<c10::NamedTensorMetaInterface> clone() const override {
    return torch::make_unique<NamedTensorMeta>(names_);
  }

  bool has_names() const;
  DimnameList names() const { return names_; }

  void set_names_(DimnameList new_names) {
    TORCH_INTERNAL_ASSERT(new_names.size() == names_.size());
    std::copy(new_names.begin(), new_names.end(), names_.begin());
  }

 private:
  std::vector<Dimname> names_;
};

namespace impl {

// Some helper functions on TensorImpl. Useful for working with names in TH.
// XXX: Ideally these would exist as methods on TensorImpl
CAFFE2_API void internal_set_names_inplace(TensorImpl* impl, optional<DimnameList> names);
CAFFE2_API optional<DimnameList> internal_get_names(TensorImpl* impl);
CAFFE2_API bool internal_is_named(TensorImpl* impl);


} // namespace impl

} // namespace at
#endif
