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
    //Note: This schema is: caffe2::layer_norm_dont_use_this_op_yet(Tensor input, int axis, float epsilon, Tensor? output = None, Tensor? output_mean = None, Tensor? output_stdev = None) -> (Tensor, Tensor, Tensor)
    c10::core::opschema::LayerNorm().schema(),
    [](Stack& stack) {
        Tensor tensor_input = std::move(stack[stack.size()-6]).toTensor();
        if (tensor_input.requires_grad()) {
          throw std::runtime_error("Autograd not yet supported for c10 ops.");
        }
        auto device = tensor_input.device();

        // unwrap inputs from variable
        torch::jit::peek(stack, 0, 6) = torch::autograd::Variable(std::move(tensor_input)).data();

        // allocate the output tensors that aren't set yet
        for (int i = 3; i < 6; ++i) {
          // TODO this should just check for isNone, not for undefined tensor. @wanchaol is working on this.
          if (torch::jit::peek(stack, i, 6).isNone() || !torch::jit::peek(stack, i, 6).toTensor().defined()) {
            torch::jit::peek(stack, i, 6) = at::empty({0}, device);
          }
        }

        // call caffe2 kernel
        c10::Dispatcher::singleton().lookup(c10::core::opschema::LayerNorm(), &stack).call(&stack);

        // wrap outputs into Variable
        for (int i = 0; i < 3; ++i) {
          torch::jit::peek(stack, i, 3) = torch::autograd::make_variable(std::move(torch::jit::peek(stack, i, 3)).toTensor(), false);
        }

        return 0;
      })
  });
}

}
}
