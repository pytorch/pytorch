#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/opschema/layer_norm.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/operator.h>
#include <ATen/core/stack.h>
#include <torch/csrc/jit/custom_operator.h>

using at::Tensor;
using c10::IValue;
using c10::ArrayRef;

namespace torch {
namespace jit {

// TODO This code is currently written specifically for LayerNorm, but it is
//      *not* the plan to have to write this manually for each operation.
//      This is just a proof of concept. To expand this to all operators,
//      we'd ideally not need any per-operator code (possibly thanks to boxing
//      or templates). If that's not possible, then we should at least offer
//      a macro that takes this burden so that we only need to write one line
//      for each operation we want to support (i.e. the macro invocation).

// TODO This currently only handles tensors with requires_grad==False correctly.
//      It should also handle autograd.

namespace {
RegisterOperators reg({
  Operator(
    "caffe2::layer_norm_dont_use_this_op_yet(Tensor input, int axis, float epsilon) -> (Tensor, Tensor, Tensor)",
    [](Stack& stack) {
        Tensor tensor_input = std::move(stack[stack.size()-3]).toTensor();
        if (tensor_input.requires_grad()) {
          throw std::runtime_error("Autograd not yet supported for c10 ops.");
        }
        auto device = tensor_input.device();
        torch::jit::peek(stack, 0, 3) = torch::autograd::Variable(std::move(tensor_input)).data();

        // push output fields as outputs to stack
        push(stack, at::empty({0}, device), at::empty({0}, device), at::empty({0}, device));

        c10::Dispatcher<c10::core::opschema::LayerNorm>::lookup(&stack).call(&stack);

        // move outputs down the stack to where the inputs were before
        for (int i = 0; i < 3; ++i) {
          torch::jit::peek(stack, i, 6) = torch::autograd::make_variable(std::move(torch::jit::peek(stack, i, 3)).toTensor(), false);
        }

        drop(stack, 3); // drop inputs

        return 0;
      })
  });
}

}
}
