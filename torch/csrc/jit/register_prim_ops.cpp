#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/fusion_compiler.h"
#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/custom_operator.h"

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
        prim::IntToFloat,
        [](Node* node) -> Operation {
          return [](Stack& stack) {
            int64_t i;
            pop(stack, i);
            push(stack, (float)i);
            return 0;
          };
        }),
    Operator(
        prim::FloatToInt,
        [](Node* node) -> Operation {
          return [](Stack& stack) {
            double d;
            pop(stack, d);
            push(stack, (int64_t)d);
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
            for (const IValue& i : last(stack, num_inputs)) {
              if (!first)
                std::cout << " ";
              first = false;
              std::cout << i;
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
          } else if (lt->getElementType()->isSubtypeOf(DynamicType::get())) {
            return [=](Stack& stack) {
              const size_t stack_size = stack.size();
              std::vector<at::Tensor> vals;
              vals.reserve(num_inputs);
              for (size_t i = stack_size - num_inputs; i < stack_size; ++i) {
                vals.push_back(std::move(stack[i]).toTensor());
              }
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
  Operator("aten::" #aten_op "(Scalar other, Tensor self) -> Tensor", [](Node* node) { \
    return [=](Stack& stack) {                                         \
      at::Scalar a;                                                    \
      at::Tensor b;                                                    \
      pop(stack, a, b);                                                \
      at::DeviceGuard guard(b);                                        \
      push(stack, reverse_exp);                                        \
      return 0;                                                        \
    };                                                                 \
  }),

// Convert an python index (which may be negative) into an index usable for a
// C++ container
int64_t normalizeIndex(int64_t idx, int64_t list_size) {
  if (idx < 0) {
    // Handle negative indexing
    idx = list_size + idx;
  }
  return idx;
}

template <typename T>
Operation listSelect(Node* node) {
  return [=](Stack& stack) {
    T list;
    int64_t idx;
    pop(stack, list, idx);
    const int64_t list_size = list->elements().size();
    const int64_t normalized_idx = normalizeIndex(idx, list_size);
    if (normalized_idx < 0 || normalized_idx >= list_size) {
      throw std::out_of_range("list index out of range");
    }

    auto element = list->elements()[normalized_idx];
    push(stack, std::move(element));
    return 0;
  };
}

template <typename T>
Operation listLen(Node* node) {
  return [=](Stack& stack) {
    T a;
    pop(stack, a);
    const int64_t size = a->elements().size();
    push(stack, size);
    return 0;
  };
}

template <typename T>
Operation listEq(Node* node) {
  return [=](Stack& stack) {
    T a;
    T b;
    pop(stack, a, b);
    if (a->elements() == b->elements()) {
      push(stack, 1);
    } else {
      push(stack, 0);
    }
    return 0;
  };
}

// Specialization for at::Tensor, since it doesn't define operator==
template <>
Operation listEq<Shared<TensorList>>(Node* node) {
  return [=](Stack& stack) {
    Shared<TensorList> a;
    Shared<TensorList> b;
    pop(stack, a, b);
    if (a->elements().size() != b->elements().size()) {
      push(stack, 0);
      return 0;
    }

    for (size_t i = 0; i < a->elements().size(); ++i) {
      const auto& a_element = a->elements()[i];
      const auto& b_element = b->elements()[i];
      // This preserves Python's semantics, which uses eq() to compare two
      // elements, then passes the result to bool().
      // see: https://docs.python.org/3.4/reference/datamodel.html#object.__ge__
      const auto cmp_result = a_element.eq(b_element);
      if (!cmp_result.is_nonzero()) {
        push(stack, 0);
        return 0;
      }
    }

    push(stack, 1);
    return 0;
  };
}

template <class TList, class TElement>
Operation listAdd(Node* node) {
  return [=](Stack& stack) {
    TList a;
    TList b;
    pop(stack, a, b);

    std::vector<TElement> ret;
    const auto total_size = a->elements().size() + b->elements().size();
    ret.reserve(total_size);
    for (const auto& a_element : a->elements()) {
      ret.push_back(a_element);
    }
    for (const auto& b_element : b->elements()) {
      ret.push_back(b_element);
    }

    push(stack, ret);
    return 0;
  };
}

template <typename TList, typename TElement>
Operation listSlice(Node* node) {
  return [](Stack& stack) {
    TList list;
    int64_t start;
    int64_t end;
    int64_t step;

    pop(stack, list, start, end, step);
    const int64_t list_size = list->elements().size();

    // clamp start and end to the bounds of the list
    const auto normalized_start =
        std::max((int64_t)0, normalizeIndex(start, list_size));
    const auto normalized_end =
        std::min(list_size, normalizeIndex(end, list_size));

    std::vector<TElement> sliced_list;
    if (normalized_end <= normalized_start) {
      // early exit if the slice is trivially empty
      push(stack, sliced_list);
      return 0;
    }

    sliced_list.reserve(normalized_end - normalized_start);

    for (auto i = normalized_start; i < normalized_end;) {
      sliced_list.push_back(list->elements()[i]);
      i += step;
    }

    push(stack, sliced_list);
    return 0;
  };
}

RegisterOperators reg2({
    Operator("aten::select(int[] a, int b) -> int", listSelect<Shared<IntList>>),
    Operator("aten::select(float[] a, int b) -> float", listSelect<Shared<DoubleList>>),
    Operator("aten::select(Tensor[] a, int b) -> Tensor", listSelect<Shared<TensorList>>),

    Operator("aten::len(int[] a) -> int", listLen<Shared<IntList>>),
    Operator("aten::len(float[] a) -> int", listLen<Shared<DoubleList>>),
    Operator("aten::len(Tensor[] a) -> int", listLen<Shared<TensorList>>),

    Operator("aten::eq(int[] a, int[] b) -> int", listEq<Shared<IntList>>),
    Operator("aten::eq(float[] a, float[] b) -> int", listEq<Shared<DoubleList>>),
    Operator("aten::eq(Tensor[] a, Tensor[] b) -> int", listEq<Shared<TensorList>>),

    Operator("aten::add(int[] a, int[] b) -> int[]", listAdd<Shared<IntList>, int64_t>),
    Operator("aten::add(float[] a, float[] b) -> float[]", listAdd<Shared<DoubleList>, double>),
    Operator("aten::add(Tensor[] a, Tensor[] b) -> Tensor[]", listAdd<Shared<TensorList>, at::Tensor>),

    Operator(
        "aten::slice(int[] l, int start, int end=9223372036854775807, int step=1) -> int[]",
        listSlice<Shared<IntList>, int64_t>),
    Operator(
        "aten::slice(float[] l, int start, int end=9223372036854775807, int step=1) -> float[]",
        listSlice<Shared<DoubleList>, double>),
    Operator(
        "aten::slice(Tensor[] l, int start, int end=9223372036854775807, int step=1) -> Tensor[]",
        listSlice<Shared<TensorList>, at::Tensor>),

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

    Operator("aten::_construct_empty_int_list() -> int[]",
        [](Node* node) -> Operation {
          return [=](Stack& stack){
            push(stack, std::vector<int64_t>());
            return 0;
        };
      }),
    Operator("aten::_construct_empty_float_list() -> float[]",
        [](Node* node) -> Operation {
          return [=](Stack& stack){
            push(stack, std::vector<double>());
            return 0;
        };
      }),
    Operator("aten::_construct_empty_tensor_list() -> Tensor[]",
        [](Node* node) -> Operation {
          return [=](Stack& stack){
            push(stack, std::vector<at::Tensor>());
            return 0;
        };
      }),
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
    Operator(
        "aten::_tensor_to_list(Tensor a) -> int[]",
        [](Node* node) {
          return [=](Stack& stack) {
            at::Tensor t;
            pop(stack, t);
            std::vector<int64_t> elems;
            for(int i = 0; i < t.size(0); i++){
              elems.push_back(*t[i].toIntData());
            }
            push(stack, jit::IntList::create(elems));
            return 0;
          };
        }),
    Operator(
        "aten::_list_to_tensor(int[] a) -> Tensor",
        [](Node* node) {
          return [=](Stack& stack) {
            std::vector<int64_t> l;
            pop(stack, l);
            auto t = torch::empty(
                {static_cast<int64_t>(l.size())}, at::dtype(at::kInt));
            for(size_t i = 0; i < l.size(); i++){
              t[i] = l[i];
            }
            push(stack, t);
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
