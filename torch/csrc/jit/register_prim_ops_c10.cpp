#include <ATen/core/op_registration/op_registration.h>
#include <ATen/ATen.h>
#include <ATen/core/stack.h>

using Stack = std::vector<c10::IValue>;
using torch::jit::peek;
using torch::jit::drop;
using torch::jit::pack;
using torch::jit::push;
using torch::jit::pop;
using at::Tensor;
using at::Scalar;
using c10::IValue;

static auto registry_prim = torch::RegisterOperators().op("aten::Int.Tensor(Tensor a) -> int",
  torch::RegisterOperators::options().catchAllKernel(
  [](at::Tensor a) -> int64_t {
    return a.item<int64_t>();
})
).op("aten::Int.bool(bool a) -> int",
  torch::RegisterOperators::options().catchAllKernel(
  [](bool b) -> int64_t {
    return static_cast<int64_t>(b);
})
).op("aten::Int.float(float a) -> int",
  torch::RegisterOperators::options().catchAllKernel(
  [](double d) -> int64_t {
    return static_cast<int64_t>(d);
})
).op("aten::Int.Scalar(Scalar a) -> int",
  torch::RegisterOperators::options().catchAllKernel(
  [](Scalar scalar) -> int64_t {
    return scalar.toInt();
})
);

