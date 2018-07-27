#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/fusion_compiler.h"
#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/operator.h"

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
        prim::TensorToNum,
        [](Node* node) -> Operation {
          if(node->output()->type() == IntType::get()) {
            return [](Stack& stack) {
              at::Tensor a;
              pop(stack, a);
              at::DeviceGuard guard(a);
              push(stack, a.toCLong());
              return 0;
            };
          } else {
            return [](Stack& stack) {
              at::Tensor a;
              pop(stack, a);
              at::DeviceGuard guard(a);
              push(stack, a.toCDouble());
              return 0;
            };
          }
        }),
    Operator(
        prim::NumToTensor,
        [](Node* node) -> Operation {
          return [](Stack& stack) {
            at::Scalar s;
            pop(stack, s);
            push(stack, autograd::make_variable(s.toTensor()));
            return 0;
          };
        }),
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
            at::Tensor input, shape;
            pop(stack, input, shape);
            shape = shape.contiguous();
            JIT_ASSERT(shape.ndimension() == 1);
            at::IntList shape_list(shape.data<int64_t>(), shape.size(0));
            push(stack, input.reshape(shape_list));
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
          return [=](Stack& stack) {
            bool result = false;
            for (const IValue& t : last(stack, num_inputs)) {
              if (std::move(t).toTensor().defined()) {
                result = true;
                break;
              }
            }
            drop(stack, num_inputs);
            stack.push_back(result);
            return 0;
          };
        }),

    Operator(
        prim::AutogradAdd,
        [](Node* node) {
          return [=](Stack& stack) {
            at::Tensor a, b;
            pop(stack, a, b);
            if (!a.defined())
              stack.push_back(b);
            else if (!b.defined())
              stack.push_back(a);
            else
              stack.push_back(a + b);
            return 0;
          };
        }),
    Operator(
        prim::ListConstruct,
        [](Node* node) -> Operation {
          size_t num_inputs = node->inputs().size();
          ListTypePtr lt = node->output()->type()->expect<ListType>();
          if(IntType::get() == lt->getElementType()) {
            return [=](Stack& stack) {
              auto inputs = peekSlice(stack, 0, num_inputs, num_inputs);
              std::vector<int64_t> vals = fmap(inputs, [](const IValue& v) {
                return v.toInt();
              });
              drop(stack, num_inputs);
              push(stack, std::move(vals));
              return 0;
            };
          } else if(FloatType::get() == lt->getElementType()) {
            return [=](Stack& stack) {
              auto inputs = peekSlice(stack, 0, num_inputs, num_inputs);
              std::vector<double> vals = fmap(inputs, [](const IValue& v) {
                return v.toDouble();
              });
              drop(stack, num_inputs);
              push(stack, std::move(vals));
              return 0;
            };
          } else {
            std::stringstream ss;
            ss << "unsupported list type: " << *lt->getElementType();
            throw std::runtime_error(ss.str());
          }
        }),
});

// define implementations for primitive number ops
#define DEFINE_GENERIC_OP(aten_op, op, float_result)                        \
  Operator(                                                                 \
      #aten_op "(int a, int b) -> int",                                     \
      [](Node* node) {                                                      \
        return [=](Stack& stack) {                                          \
          int64_t a, b;                                                     \
          pop(stack, a, b);                                                 \
          push(stack, op);                                                  \
          return 0;                                                         \
        };                                                                  \
      }),                                                                   \
  Operator(                                                                 \
      #aten_op "(float a, float b) -> " #float_result, [](Node* node) {     \
        return [=](Stack& stack) {                                          \
          double a, b;                                                      \
          pop(stack, a, b);                                                 \
          push(stack, op);                                                  \
          return 0;                                                         \
        };                                                                  \
      }),

#define DEFINE_INT_OP(aten_op, op)                            \
  Operator(#aten_op "(int a, int b) -> int", [](Node* node) { \
    return [=](Stack& stack) {                                \
      int64_t a, b;                                           \
      pop(stack, a, b);                                       \
      push(stack, op);                                        \
      return 0;                                               \
    };                                                        \
  }),

#define DEFINE_BINARY_OP(aten_op, op) DEFINE_GENERIC_OP(aten_op, op, float)
#define DEFINE_COMPARISON_OP(aten_op, op) DEFINE_GENERIC_OP(aten_op, op, int)

// define helpers for where aten is missing scalar overloads
// note: it would be better to define these in a standard library as
// script functions and have the compiler substitute them in
// however, we need to add type annotations to the parser in order for us
// to move them there.
// e.g. s + t ==> t + s
// e.g. s - d == -d + s

#define DEFINE_ST_OP(aten_op, reverse_exp)                             \
  Operator("aten::" #aten_op "(Scalar a, Tensor b) -> Tensor", [](Node* node) { \
    return [=](Stack& stack) {                                         \
      at::Scalar a;                                                    \
      at::Tensor b;                                                    \
      pop(stack, a, b);                                                \
      at::DeviceGuard guard(b);                                        \
      push(stack, reverse_exp);                                        \
      return 0;                                                        \
    };                                                                 \
  }),

RegisterOperators reg2({
    DEFINE_BINARY_OP(aten::add, a + b)
    DEFINE_BINARY_OP(aten::sub, a - b)
    DEFINE_BINARY_OP(aten::mul, a * b)
    DEFINE_BINARY_OP(aten::div, a / b)
    DEFINE_BINARY_OP(aten::pow, static_cast<decltype(a)>(pow(a, b)))

    DEFINE_COMPARISON_OP(aten::ne, a != b)
    DEFINE_COMPARISON_OP(aten::eq, a == b)
    DEFINE_COMPARISON_OP(aten::lt, a < b)
    DEFINE_COMPARISON_OP(aten::gt, a > b)
    DEFINE_COMPARISON_OP(aten::le, a <= b)
    DEFINE_COMPARISON_OP(aten::ge, a >= b)

    DEFINE_INT_OP(aten::__and__, a&& b)
    DEFINE_INT_OP(aten::__or__, a || b)

    Operator(
        "aten::neg(int a) -> int",
        [](Node* node) {
          return [=](Stack& stack) {
            push(stack, -pop(stack).toInt());
            return 0;
          };
        }),
    Operator(
        "aten::neg(float a) -> float",
        [](Node* node) {
          return [=](Stack& stack) {
            push(stack, -pop(stack).toDouble());
            return 0;
          };
        }),
    Operator(
        "aten::__not__(int a) -> int",
        [](Node* node) {
          return [=](Stack& stack) {
            push(stack, !pop(stack).toInt());
            return 0;
          };
        }),

    // commutative
    DEFINE_ST_OP(mul, at::mul(b, a))
    DEFINE_ST_OP(add, at::add(b, a))
    DEFINE_ST_OP(ne, at::ne(b, a))
    DEFINE_ST_OP(eq, at::eq(b, a))

    // comparisons, reverse the condition
    DEFINE_ST_OP(lt, b > a)
    DEFINE_ST_OP(le, b >= a)
    DEFINE_ST_OP(gt, b < a)
    DEFINE_ST_OP(ge, b <= a)

    // rsub
    DEFINE_ST_OP(sub, at::add(b.neg(), a))
    // rdiv
    DEFINE_ST_OP(div, at::mul(at::reciprocal(b), a))
});
}}} // torch::jit::anon
