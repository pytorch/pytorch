#pragma once

#include <ATen/core/Tensor.h>
#include <torch/csrc/python_headers.h>

#include <torch/csrc/utils/python_arg_parser.h>

namespace torch::autograd::utils {

// The parameters allow_copy and allow_round are to accept copy and round for
// Tensor.to (and by proxy PackedSequences.to) but not nn.Module.to.
inline std::tuple<
    std::optional<at::Device>,
    std::optional<at::ScalarType>,
    bool,
    bool,
    std::optional<at::MemoryFormat>,
    std::optional<std::string_view>>
parse_to_conversion(PythonArgs& r, bool allow_copy, bool allow_round) {
  if (r.idx == 0) {
    TORCH_CHECK(
        allow_copy || r.isNone(3), ".to() does not accept copy argument");
    TORCH_CHECK(
        allow_round || r.isNone(5), ".to() does not accept round argument");
    return std::make_tuple(
        r.deviceOptional(0),
        r.scalartypeOptional(1),
        r.toBool(2),
        r.toBool(3),
        r.memoryformatOptional(4),
        r.stringViewOptional(5));
  } else if (r.idx == 1) {
    TORCH_CHECK(
        allow_copy || r.isNone(2), ".to() does not accept copy argument");
    TORCH_CHECK(
        allow_round || r.isNone(4), ".to() does not accept round argument");
    return std::make_tuple(
        std::nullopt,
        r.scalartype(0),
        r.toBool(1),
        r.toBool(2),
        r.memoryformatOptional(3),
        r.stringViewOptional(4));
  } else {
    auto tensor = r.tensor(0);
    TORCH_CHECK(
        allow_copy || r.isNone(2), ".to() does not accept copy argument");
    TORCH_CHECK(
        allow_round || r.isNone(4), ".to() does not accept round argument");
    return std::make_tuple(
        tensor.device(),
        tensor.scalar_type(),
        r.toBool(1),
        r.toBool(2),
        r.memoryformatOptional(3),
        r.stringViewOptional(4));
  }
}
} // namespace torch::autograd::utils
