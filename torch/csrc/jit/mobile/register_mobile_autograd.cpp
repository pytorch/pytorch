#include <ATen/core/op_registration/op_registration.h>
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <ATen/TypeDefault.h>
#include <torch/csrc/autograd/function.h>

using Stack = std::vector<c10::IValue>;
using torch::jit::peek;
using torch::jit::drop;
using torch::jit::pack;
using torch::jit::push;
using torch::jit::pop;
using at::Tensor;
using at::Scalar;
using namespace torch::autograd;
using namespace c10;

namespace torch {
namespace autograd {
namespace VariableType {
Tensor mul_Tensor(const Tensor &self, const Tensor &other);
Tensor add_Scalar(const Tensor &self, Scalar other, Scalar alpha);
Tensor conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups);
Tensor view(const Tensor & self, IntArrayRef size);
Tensor log_softmax_int(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype);
Tensor dropout(const Tensor & input, double p, bool train);
Tensor feature_dropout(const Tensor & input, double p, bool train);
Tensor max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode);
Tensor relu(const Tensor & self);
Tensor t(const Tensor & self);
Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha);
}
}
}

namespace {
at::Tensor toOptionalTensor(const c10::IValue& v) {
  if (v.isNone()) {
    return at::Tensor();
  }
  return v.toTensor();
}

void conv2d_kernel(const c10::OperatorHandle& op, Stack* stack) {
  auto input = std::move(peek(*stack, 0, 7)).toTensor();
  auto weight = (std::move(peek(*stack, 1, 7))).toTensor();
  auto bias = toOptionalTensor((std::move(peek(*stack, 2, 7))));
  RECORD_FUNCTION("conv2d", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  auto result_ = VariableType::conv2d(
      input,
      weight,
      bias,
      (std::move(peek(*stack, 3, 7))).toIntVector(),
      (std::move(peek(*stack, 4, 7))).toIntVector(),
      (std::move(peek(*stack, 5, 7))).toIntVector(),
      (std::move(peek(*stack, 6, 7))).toInt()
      );
  drop(*stack, 7);
  pack(*stack, std::move(result_));
}

void view_kernel(const c10::OperatorHandle& op, Stack* stack) {
  auto self = (std::move(peek(*stack, 0, 2))).toTensor();
  auto size = (std::move(peek(*stack, 1, 2))).toIntVector();
  auto result_ = torch::autograd::VariableType::view(self, size);
  drop(*stack, 2);
  pack(*stack, std::move(result_));
}

void log_softmax_kernel(const c10::OperatorHandle& op, Stack* stack) {
  auto self = (std::move(peek(*stack, 0, 3))).toTensor();
  auto dim = (std::move(peek(*stack, 1, 3))).toInt();
  auto dtype = (std::move(peek(*stack, 2, 3))).toOptional<c10::ScalarType>();
  auto result_ = torch::autograd::VariableType::log_softmax_int(self, dim, dtype);
  drop(*stack, 3);
  pack(*stack, std::move(result_));
}

// NB! This is _aten, not aten!!!
static auto registry = torch::import("_aten")
  .impl("add.Scalar",       torch::dispatch_autograd(torch::autograd::VariableType::add_Scalar))
  .impl("mul.Tensor",       torch::dispatch_autograd(torch::autograd::VariableType::mul_Tensor))
  .impl("conv2d",           torch::dispatch_autograd(CppFunction::makeFromBoxedFunction<conv2d_kernel>()))
  .impl("dropout",          torch::dispatch_autograd(VariableType::dropout))
  .impl("feature_dropout",  torch::dispatch_autograd(VariableType::feature_dropout))
  .impl("log_softmax.int",  torch::dispatch_autograd(CppFunction::makeFromBoxedFunction<log_softmax_kernel>()))
  .impl("max_pool2d",       torch::dispatch_autograd(
      [](const Tensor & self, c10::List<int64_t> kernel_size, c10::List<int64_t> stride,
         c10::List<int64_t> padding, c10::List<int64_t> dilation, bool ceil_mode=false) {
            return VariableType::max_pool2d(self, kernel_size.vec(), stride.vec(), padding.vec(), dilation.vec(), ceil_mode);
      }))
  .impl("relu",             torch::dispatch_autograd(VariableType::relu))
  .impl("view",             torch::dispatch_autograd(CppFunction::makeFromBoxedFunction<view_kernel>()))
  .impl("t",                torch::dispatch_autograd(VariableType::t))
  .impl("addmm",            torch::dispatch_autograd(VariableType::addmm))
;
} // anonymous namespace
