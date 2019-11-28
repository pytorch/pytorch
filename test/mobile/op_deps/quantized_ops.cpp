#include "quantized_ops.h"

#include <iostream>

#include <c10/core/TensorOptions.h>
#include <ATen/core/op_registration/op_registration.h>

// This file simulates some irregular op registration/invocation patterns for
// quantized operators which are not covered by aten codegen.

namespace at {

namespace {

template <bool ReLUFused>
Tensor _add_out(Tensor& out, const Tensor& self, const Tensor& other);

template <>
Tensor _add_out<false>(Tensor& out, const Tensor& self, const Tensor& other) {
  const auto kName = "quantized::t_helper1";
  callOp(kName, "", self);
  return out;
}

template <>
Tensor _add_out<true>(Tensor& out, const Tensor& self, const Tensor& other) {
  const auto kName = "quantized::t_helper2";
  callOp(kName, "", self);
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

template <const char* str>
class QHelper final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qa) {
    std::cout << str << std::endl;
    return qa;
  }
};

static char helper1[] = "quantized helper1";
static char helper2[] = "quantized helper2";

static auto registry = c10::RegisterOperators()
.op("quantized::t_add(Tensor qa, Tensor qb, float scale, int zero_point)"
     "-> Tensor qc",
    c10::RegisterOperators::options()
      .catchAllKernel<QAdd</*ReLUFused=*/false>>())
.op("quantized::t_add_relu(Tensor qa, Tensor qb, float scale, int zero_point)"
     "-> Tensor qc",
    c10::RegisterOperators::options()
      .catchAllKernel<QAdd</*ReLUFused=*/true>>())
.op("quantized::t_helper1(Tensor qa) -> Tensor",
    c10::RegisterOperators::options()
      .catchAllKernel<QHelper<helper1>>())
.op("quantized::t_helper2(Tensor qa) -> Tensor",
    c10::RegisterOperators::options()
      .catchAllKernel<QHelper<helper2>>());

} // namespace

} // namespace at
