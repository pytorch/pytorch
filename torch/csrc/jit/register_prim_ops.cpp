#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/fusion_compiler.h"
#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/tensor_conversions.h"
#include "torch/csrc/variable_tensor_functions.h"

#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

namespace {

Operation noop(Node* n) {
  return [](Stack& stack) { return 0; };
}

RegisterOperators reg({

    Operator(
        prim::FusionGroup,
        [](Node* node) {
          auto fusion_fn = sharedFusionCompiler().getOrCompile(node);
          auto num_inputs = node->inputs().size();
          return [fusion_fn, num_inputs](Stack& stack) {
            autograd::profiler::RecordFunction record("FusionGroup");
            std::vector<at::Tensor> toutputs;
            // TODO: have fusion_fn work off of a stack as well
            auto tinputs = fmap(last(stack, num_inputs), [](const IValue& v) {
              return v.toTensor();
            });
            fusion_fn->launch(tinputs, toutputs);
            drop(stack, num_inputs);
            stack.insert(stack.end(), toutputs.begin(), toutputs.end());
            return 0;
          };
        }),

    Operator(
        prim::Constant,
        [](Node* node) {
          auto t = autograd::make_variable(node->t(attr::value));
          return [t](Stack& stack) {
            stack.push_back(t);
            return 0;
          };
        }),

    Operator(prim::NumToTensor, noop),
    Operator(prim::TensorToNum, noop),
    Operator(
        prim::Undefined,
        [](Node* node) {
          return [](Stack& stack) {
            stack.push_back(at::Tensor());
            return 0;
          };
        }),
    Operator(
        prim::Print,
        [](Node* node) {
          size_t num_inputs = node->inputs().size();
          return [num_inputs](Stack& stack) {
            bool first = true;
            for (const IValue& i_ : last(stack, num_inputs)) {
              auto i = i_.toTensor();
              if (!first)
                std::cout << " ";
              first = false;
              if (auto tensor_impl = dynamic_cast<at::TensorImpl*>(i.get())) {
                std::cout << at::Tensor(tensor_impl, true);
              } else if (!i.defined()) {
                std::cout << "<undefined tensor>";
              } else {
                auto& r = *i.get();
                std::cout << "<" << typeid(r).name() << " at " << i << ">";
              }
            }
            drop(stack, num_inputs);
            std::cout << std::endl;
            return 0;
          };
        }),
    // Load x, y
    // loads values from registers onto the stack, the actual callback does
    // nothing since the stack manipulation is already encoded in inst.inputs
    // and inst.outputs
    Operator(prim::Load, noop),
    // x, y = Store
    // stores vales from stack into registers, the actual callback does
    // nothing since the stack manipulation is already encoded in inst.inputs
    // and inst.outputs
    Operator(prim::Store, noop),

    Operator(
        prim::Drop,
        [](Node* node) {
          auto N = node->inputs().size();
          return [=](Stack& stack) {
            drop(stack, N);
            return 0;
          };
        }),
    Operator(
        onnx::Reshape,
        [](Node* node) {
          return [=](Stack& stack) {
            auto shape = pop(stack).toTensor().contiguous();
            auto input = pop(stack).toTensor();
            JIT_ASSERT(shape.ndimension() == 1);
            at::IntList shape_list(shape.data<int64_t>(), shape.size(0));
            stack.push_back(input.reshape(shape_list));
            return 0;
          };
        }),
    Operator(
        onnx::Shape,
        [](Node* node) {
          return [=](Stack& stack) {
            auto t = pop(stack).toTensor();
            at::IntList sizes = t.sizes();
            auto sizes_tensor = torch::empty(
                {static_cast<int64_t>(sizes.size())}, at::dtype(at::kLong));
            auto accessor = sizes_tensor.accessor<int64_t, 1>();
            for (size_t i = 0; i < sizes.size(); ++i) {
              accessor[i] = sizes[i];
            }
            stack.push_back(sizes_tensor);
            return 0;
          };
        }),

    Operator(
        prim::AnyDefined,
        [](Node* node) {
          size_t num_inputs = node->inputs().size();
          auto true_ = at::full({}, 1, at::kLong);
          auto false_ = at::full({}, 0, at::kLong);
          return [=](Stack& stack) {
            bool result = false;
            for (const IValue& t : last(stack, num_inputs)) {
              if (std::move(t).toTensor().defined()) {
                result = true;
                break;
              }
            }
            drop(stack, num_inputs);
            stack.push_back(result ? true_ : false_);
            return 0;
          };
        }),

    Operator(
        prim::AutogradAdd,
        [](Node* node) {
          return [=](Stack& stack) {
            auto a = pop(stack).toTensor();
            auto b = pop(stack).toTensor();
            if (!a.defined())
              stack.push_back(b);
            else if (!b.defined())
              stack.push_back(a);
            else
              stack.push_back(a + b);
            return 0;
          };
        }),
});
}}} // torch::jit::anon
