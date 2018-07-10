#pragma once

#include "torch/csrc/python_headers.h"
#include <ATen/ATen.h>

#include "torch/csrc/utils/python_arg_parser.h"

namespace torch { namespace autograd { namespace utils {

inline std::tuple<at::optional<at::Device>, at::optional<at::ScalarType>, bool>
parse_to_conversion(PyObject *args, PyObject *kwargs) {
  static PythonArgParser parser({
    "to(Device device=None, ScalarType dtype=None, bool non_blocking=False)",
    "to(ScalarType dtype, bool non_blocking=False)",
    "to(Tensor tensor, bool non_blocking=False)",
  });
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return std::make_tuple(r.deviceOptional(0), r.scalartypeOptional(1), r.toBool(2));
  } else if (r.idx == 1) {
    return std::make_tuple(at::nullopt, r.scalartype(0), r.toBool(1));
  } else {
    auto tensor = r.tensor(0);
    return std::make_tuple(
      torch::tensors::getDevice(tensor),
      tensor.type().scalarType(),
      r.toBool(1)
    );
  }
}

}}} // namespace torch::autograd::utils
