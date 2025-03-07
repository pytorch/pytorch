#include <pybind11/pybind11.h>

#include "torch/csrc/runtime/package/pt2_archive_constants.h"

PYBIND11_MODULE(pt2_archive_constants_pybind, m) {
  for (const auto& entry : torch::runtime::archive_spec::kAllConstants) {
    m.attr(entry.first) = entry.second;
  }
}
