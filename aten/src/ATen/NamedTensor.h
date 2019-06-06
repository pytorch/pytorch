#pragma once
#ifdef NAMEDTENSOR_ENABLED

#include <ATen/Dimname.h>
#include <c10/core/TensorImpl.h>

namespace at {

// TensorImpl has a unique_ptr<NamedTensorMetaInterface> field. Ideally we would
// just put optional<vector<Dimname>> into TensorImpl, but the problem with that is
// that c10::Symbol isn't actually a part of the c10 lib (where TensorImpl is).
// In the long term, we should decouple c10::Symbol from aten and toss it into c10.
struct CAFFE2_API NamedTensorMeta : public c10::NamedTensorMetaInterface {
  std::vector<Dimname> names;

  explicit NamedTensorMeta(int64_t num_names)
    : names(std::vector<Dimname>(num_names, Dimname::wildcard())) {}

  explicit NamedTensorMeta(std::vector<Dimname> names)
    : names(names) {}

  bool has_names() const;
};

}
#endif
