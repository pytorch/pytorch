#pragma once

#include <torch/csrc/python_headers.h>
#include <ATen/ATen.h>

#include <torch/csrc/utils/python_arg_parser.h>

namespace torch { namespace autograd { namespace utils {

// The parameter allow_copy is to accept copy for Tensor.to (and by proxy
// PackedSequences.to) but not nn.Module.to.
inline std::tuple<c10::optional<at::Device>, c10::optional<at::ScalarType>, bool, bool>
  parse_to_conversion(PyObject *args, PyObject *kwargs, bool allow_copy) {
  static PythonArgParser parser({
    "to(Device device=None, ScalarType dtype=None, bool non_blocking=False, bool copy=False)",
    "to(ScalarType dtype, bool non_blocking=False, bool copy=False)",
    "to(Tensor tensor, bool non_blocking=False, bool copy=False)",
  });
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (!allow_copy && !r.isNone(3))
      throw std::runtime_error(".to() does not accept copy argument");
    return std::make_tuple(r.deviceOptional(0), r.scalartypeOptional(1), r.toBool(2), r.toBool(3));
  } else if (r.idx == 1) {
    if (!allow_copy && !r.isNone(2))
      throw std::runtime_error(".to() does not accept copy argument");
    return std::make_tuple(c10::nullopt, r.scalartype(0), r.toBool(1), r.toBool(2));
  } else {
    auto tensor = r.tensor(0);
    if (!allow_copy && !r.isNone(2))
      throw std::runtime_error(".to() does not accept copy argument");
    return std::make_tuple(
      tensor.device(),
      tensor.scalar_type(),
      r.toBool(1),
      r.toBool(2)
    );
  }
}
}}} // namespace torch::autograd::utils
