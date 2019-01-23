#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/opschema/layer_norm.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/stack.h>
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
        ArrayRef<IValue> inputs = last(stack, 3);
        Tensor input = std::move(inputs[0]).toTensor();
        const IValue& axis = inputs[1];
        const IValue& epsilon = inputs[2];
        drop(stack, 3);

        if (input.requires_grad()) {
          throw std::runtime_error("Autograd not yet supported for c10 ops.");
        }

        Tensor c10_output(at::empty({0}, input.device()));
        Tensor c10_output_mean(at::empty({0}, input.device()));
        Tensor c10_output_stdev(at::empty({0}, input.device()));

        c10::intrusive_ptr<caffe2::Blob> cache = c10::make_intrusive<caffe2::Blob>();
        cache->GetMutable<c10::core::opschema::LayerNorm::Cache>(); // initialize cache
        // TODO remove std::array for args here, instead pass through inputs directly as ArrayRef
        std::array<c10::IValue, 7> args{
          IValue(torch::autograd::Variable(std::move(input)).data()),
          IValue(c10_output),
          IValue(c10_output_mean),
          IValue(c10_output_stdev),
          axis,
          epsilon,
          IValue(cache)
        };
        c10::Dispatcher<c10::core::opschema::LayerNorm>::lookup(args).call(args);
        push(stack,
          torch::autograd::make_variable(at::Tensor(std::move(c10_output)), false),
          torch::autograd::make_variable(at::Tensor(std::move(c10_output_mean)), false),
          torch::autograd::make_variable(at::Tensor(std::move(c10_output_stdev)), false)
        );
        return 0;
      })
  });
}

}
}
