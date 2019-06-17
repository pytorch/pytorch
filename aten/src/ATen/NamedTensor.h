#pragma once
#ifdef NAMEDTENSOR_ENABLED

#include <ATen/Dimname.h>
#include <c10/core/TensorImpl.h>
#include <torch/csrc/utils/memory.h>

namespace at {

// TensorImpl has a unique_ptr<NamedTensorMetaInterface> field. Ideally we would
// just put optional<vector<Dimname>> into TensorImpl, but the problem with that is
// that c10::Symbol isn't actually a part of the c10 lib (where TensorImpl is).
// In the long term, we should decouple c10::Symbol from aten and toss it into c10.
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

}
#endif
