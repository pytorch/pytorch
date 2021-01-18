#include "quantized_ops.h"

#include <iostream>

#include <c10/core/TensorOptions.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/op_registration.h>

// This file simulates some irregular op registration/invocation patterns for
// quantized operators which are not covered by aten codegen.

namespace at {

namespace {

template <bool ReLUFused>
Tensor _add_out(Tensor& out, const Tensor& self, const Tensor& other);

template <>
Tensor _add_out<false>(Tensor& out, const Tensor& self, const Tensor& other) {
  constexpr auto kName = "quantized::t_helper1";
  static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(kName, "").typed<Tensor(Tensor)>();;
  op.call(self);
  return out;
}

template <>
Tensor _add_out<true>(Tensor& out, const Tensor& self, const Tensor& other) {
  constexpr auto kName = "quantized::t_helper2";
  static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(kName, "").typed<Tensor(Tensor)>();
  op.call(self);
  return out;
}

template <bool ReLUFused = false>
class QAdd final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
    std::cout << "QAdd with ReLUFused = " << ReLUFused << std::endl;
    return _add_out<ReLUFused>(qa, qa, qb); // hack
  }
};

template <const char* opName, const char* callOpName>
Tensor QHelper(Tensor qa) {
  std::cout << "Op: " << opName << std::endl;
  if (callOpName != nullptr) {
    std::cout << "Call op: " << callOpName << std::endl;
    static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(callOpName, "").typed<Tensor(Tensor)>();
    op.call(qa);
  }
  return qa;
}

constexpr char helper1[] = "quantized::t_helper1";
constexpr char helper2[] = "quantized::t_helper2";
constexpr char helper3[] = "quantized::t_helper3";
constexpr char helper4[] = "quantized::t_helper4";

static auto registry = c10::RegisterOperators()
.op("quantized::t_add(Tensor qa, Tensor qb, float scale, int zero_point)"
     "-> Tensor qc",
    c10::RegisterOperators::options()
      .catchAllKernel<QAdd</*ReLUFused=*/false>>())
.op("quantized::t_add_relu(Tensor qa, Tensor qb, float scale, int zero_point)"
     "-> Tensor qc",
    c10::RegisterOperators::options()
      .catchAllKernel<QAdd</*ReLUFused=*/true>>())
.op("quantized::t_helper1(Tensor qa) -> Tensor", &QHelper<helper1, helper3>)
.op("quantized::t_helper2(Tensor qa) -> Tensor", &QHelper<helper2, helper4>)
.op("quantized::t_helper3(Tensor qa) -> Tensor", &QHelper<helper3, nullptr>)
.op("quantized::t_helper4(Tensor qa) -> Tensor", &QHelper<helper4, nullptr>);

} // namespace

} // namespace at
