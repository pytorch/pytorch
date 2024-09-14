#pragma once

#include <ATen/core/Tensor.h>
#include <torch/csrc/python_headers.h>

#include <torch/csrc/utils/python_arg_parser.h>

namespace torch::autograd::utils {

// The parameter allow_copy is to accept copy for Tensor.to (and by proxy
// PackedSequences.to) but not nn.Module.to.
inline std::tuple<
    std::optional<at::Device>,
    std::optional<at::ScalarType>,
    bool,
    bool,
    std::optional<at::MemoryFormat>>
parse_to_conversion(PythonArgs& r, bool allow_copy) {
  if (r.idx == 0) {
    if (!allow_copy && !r.isNone(3))
      throw std::runtime_error(".to() does not accept copy argument");
    return std::make_tuple(
        r.deviceOptional(0),
        r.scalartypeOptional(1),
        r.toBool(2),
        r.toBool(3),
        r.memoryformatOptional(4));
  } else if (r.idx == 1) {
    if (!allow_copy && !r.isNone(2))
      throw std::runtime_error(".to() does not accept copy argument");
    return std::make_tuple(
        std::nullopt,
        r.scalartype(0),
        r.toBool(1),
        r.toBool(2),
        r.memoryformatOptional(3));
  } else {
    auto tensor = r.tensor(0);
    if (!allow_copy && !r.isNone(2))
      throw std::runtime_error(".to() does not accept copy argument");
    return std::make_tuple(
        tensor.device(),
        tensor.scalar_type(),
        r.toBool(1),
        r.toBool(2),
        r.memoryformatOptional(3));
  }
}
} // namespace torch::autograd::utils
