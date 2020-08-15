#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/stack.h>
#include <c10/util/string_utils.h>
#include <torch/library.h>

using Stack = std::vector<c10::IValue>;
using at::Scalar;
using at::Tensor;
using c10::IValue;
using torch::jit::drop;
using torch::jit::pack;
using torch::jit::peek;
using torch::jit::pop;
using torch::jit::push;

// Implementations located in torch/csrc/jit/runtime/register_prim_ops_c10.cpp
TORCH_LIBRARY_IMPL(aten, CatchAll, m) {
  m.impl("Int.Tensor", [](at::Tensor a) { return a.item<int64_t>(); });

  m.impl("Int.bool", [](bool b) { return static_cast<int64_t>(b); });

  m.impl("Int.float", [](double d) { return static_cast<int64_t>(d); });

  m.impl("Int.Scalar", [](Scalar scalar) {
    return static_cast<int64_t>(scalar.toInt());
  });

  m.impl("Int.str", [](const std::string& str) {
    std::string::size_type sz;
    int64_t val = static_cast<int64_t>(c10::stoll(str, &sz));
    if (sz != str.size()) {
      std::stringstream error_str;
      error_str << "invalid literal for int() "
                << "with base 10: '" << str << "'";
      throw std::runtime_error(error_str.str());
    }
    return val;
  });
}
