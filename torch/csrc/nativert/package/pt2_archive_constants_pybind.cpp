#include <pybind11/pybind11.h>

#include "torch/csrc/nativert/package/pt2_archive_constants.h"

namespace torch::nativert {
void initPt2ArchiveConstantsPybind(pybind11::module& m) {
  for (const auto& entry : torch::nativert::archive_spec::kAllConstants) {
    m.attr(entry.first) = entry.second;
  }
}
} // namespace torch::nativert

// TODO Remove this once we fully migrate to OSS build.
#ifdef FBCODE_CAFFE2
PYBIND11_MODULE(pt2_archive_constants_pybind, m) {
  torch::nativert::initPt2ArchiveConstantsPybind(m);
}
#endif
