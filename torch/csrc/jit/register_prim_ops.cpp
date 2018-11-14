#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/custom_operator.h"
#include "torch/csrc/jit/script/jit_exception.h"

#include "torch/csrc/variable_tensor_functions.h"

#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

namespace {

Operation noop(const Node* n) {
  return [](Stack& stack) { return 0; };
}

// using the rules from python_arg_parser FunctionParameter::check
// tensor cannot have grad set, tensor must be 0 dim,
// and if the dest is an int the source must be integral type
void checkImplicitTensorToNum(at::Tensor t, bool toInt) {
  if (autograd::as_variable_ref(t).requires_grad()) {
    throw std::runtime_error("Cannot input a tensor that requires grad as a scalar argument");
  }
  if (t.sizes().size() != 0) {
    throw std::runtime_error("Cannot input a tensor of dimension other than 0 as a scalar argument");
  }
  if (toInt && !isIntegralType(autograd::as_variable_ref(t).data().type().scalarType())) {
    std::stringstream ss;
    ss << "Cannot input a tensor of type " << t.type().scalarType() << " as an integral argument";
    throw std::runtime_error(ss.str());
  }
}

RegisterOperators reg({
    Operator(
        prim::FusionGroup,
        [](const Node* node) {
          const auto key = registerFusion(node);
          return [key](Stack& stack) {
            autograd::profiler::RecordFunction record("FusionGroup");
            runFusion(key, stack);
            return 0;
          };
        }),
    Operator(
        "prim::TensorToBool(Tensor a) -> bool",
        [](const Node* node) -> Operation {
          return [](Stack& stack) {
            at::Tensor a;
            pop(stack, a);
            at::OptionalDeviceGuard guard(device_of(a));
            push(stack, a.item<int64_t>() != 0);
            return 0;
          };
        }),
    Operator(
        "prim::TensorToNum(Tensor a) -> Scalar",
        [](const Node* node) -> Operation {
          if(node->output()->type() == IntType::get()) {
            return [](Stack& stack) {
              at::Tensor a;
              pop(stack, a);
              at::OptionalDeviceGuard guard(device_of(a));
              push(stack, a.item<int64_t>());
              return 0;
            };
          } else {
            return [](Stack& stack) {
              at::Tensor a;
              pop(stack, a);
              at::OptionalDeviceGuard guard(device_of(a));
              push(stack, a.item<double>());
              return 0;
            };
          }
        }),
    Operator(
        "prim::ImplicitTensorToNum(Tensor a) -> Scalar",
        [](const Node* node) -> Operation {
          if(node->output()->type() == IntType::get()) {
            return [](Stack& stack) {
              at::Tensor a;
              pop(stack, a);
              checkImplicitTensorToNum(a, /*to int*/true);
              at::OptionalDeviceGuard guard(device_of(a));
              push(stack, a.item<int64_t>());
              return 0;
            };
          } else {
            return [](Stack& stack) {
              at::Tensor a;
              pop(stack, a);
              checkImplicitTensorToNum(a, /*to int*/false);
              at::OptionalDeviceGuard guard(device_of(a));
              push(stack, a.item<double>());
              return 0;
            };
          }
        }),
    Operator(
        "prim::NumToTensor(Scalar a) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack& stack) {
            at::Scalar s;
            pop(stack, s);
            push(stack, autograd::make_variable(at::scalar_to_tensor(s)));
            return 0;
          };
        }),
    Operator(
        "prim::BoolToTensor(bool a) -> Tensor",
        [](const Node* node) -> Operation {
          return [](Stack& stack) {
            bool b;
            pop(stack, b);
            push(
                stack,
                autograd::make_variable(at::scalar_to_tensor(b)));
            return 0;
          };
        }),
    Operator(
        "prim::IntToFloat(int a) -> float",
        [](const Node* node) -> Operation {
          return [](Stack& stack) {
            int64_t i;
            pop(stack, i);
            push(stack, (float)i);
            return 0;
          };
        }),
    Operator(
        "prim::FloatToInt(float a) -> int",
        [](const Node* node) -> Operation {
          return [](Stack& stack) {
            double d;
            pop(stack, d);
            push(stack, (int64_t)d);
            return 0;
          };
        }),
    Operator(
        "prim::StringToFloat(str a) -> float",
        [](const Node* node) -> Operation {
          return [](Stack& stack) {
            auto s = pop(stack).toString();
            if (s->string() == "inf")
              push(stack, std::numeric_limits<double>::infinity());
            else if (s->string() == "-inf")
              push(stack, -std::numeric_limits<double>::infinity());
            else
              AT_ERROR(
                  "Only 'inf' or '-inf' can be cast to a float, but got '",
                  s->string(),
                  "'");
            return 0;
          };
        }),
    Operator(
        "prim::device(Tensor a) -> int[]",
        [](const Node* node) -> Operation {
          return [](Stack& stack) {
            at::Tensor a;
            pop(stack, a);
            push(stack, std::vector<int64_t>({static_cast<int64_t>(a.device().type()),
                                              a.device().index()}));
            return 0;
          };
        }),
    Operator(
        "prim::dtype(Tensor a) -> int",
        [](const Node* node) -> Operation {
          return [](Stack& stack) {
            at::Tensor a;
            pop(stack, a);
            push(stack, static_cast<int64_t>(a.scalar_type()));
            return 0;
          };
        }),
    Operator(
        "prim::shape(Tensor a) -> int[]",
        [](const Node* node) -> Operation {
          return [](Stack& stack) {
            at::Tensor a;
            pop(stack, a);
            push(stack, a.sizes());
            return 0;
          };
        }),
    Operator(
        prim::Undefined,
        [](const Node* node) {
          return [](Stack& stack) {
            stack.emplace_back(at::Tensor());
            return 0;
          };
        }),
    Operator(
      prim::None,
      [](const Node* node) {
        return [](Stack& stack) {
          stack.emplace_back(IValue());
          return 0;
        };
      }),
    Operator(
        prim::NoneGenerator,
        [](const Node* node) {
          return [](Stack& stack) {
            stack.emplace_back();
            return 0;
          };
        }),
    Operator(
        prim::Print,
        [](const Node* node) {
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
    Operator(
        "prim::RaiseException(str msg) -> ()",
        [](const Node* node) -> Operation {
          return [](Stack& stack) {
            throw JITException(pop(stack).toStringRef());
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
        [](const Node* node) {
          auto N = node->inputs().size();
          return [=](Stack& stack) {
            drop(stack, N);
            return 0;
          };
        }),
    Operator(
        onnx::Reshape,
        [](const Node* node) {
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
        [](const Node* node) {
          return [=](Stack& stack) {
            auto t = pop(stack).toTensor();
            at::IntList sizes = t.sizes();
            auto sizes_tensor = torch::empty(
                {static_cast<int64_t>(sizes.size())}, at::dtype(at::kLong));
            auto accessor = sizes_tensor.accessor<int64_t, 1>();
            for (size_t i = 0; i < sizes.size(); ++i) {
              accessor[i] = sizes[i];
            }
            stack.emplace_back(sizes_tensor);
            return 0;
          };
        }),

    Operator(
        prim::AnyDefined,
        [](const Node* node) {
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
            stack.emplace_back(result);
            return 0;
          };
        }),

    Operator(
        prim::AutogradAdd,
        [](const Node* node) {
          return [=](Stack& stack) {
            at::Tensor a, b;
            pop(stack, a, b);
            if (!a.defined())
              stack.emplace_back(b);
            else if (!b.defined())
              stack.emplace_back(a);
            else
              stack.emplace_back(a + b);
            return 0;
          };
        }),
    Operator(
        prim::TupleUnpack,
        [](const Node* node) {
          size_t num_elems = node->outputs().size();
          return [=](Stack& stack) {
            auto t = pop(stack).toTuple();
            const auto & elems = t->elements();
            if (elems.size() != num_elems) {
              AT_ERROR("Expected a tuple of ", num_elems, " elements, but got ", elems.size());
            }
            stack.insert(stack.end(), elems.begin(), elems.end());
            return 0;
          };
        }),
    Operator(
        prim::TupleSlice,
        [](const Node* node) {
          int64_t beg_ind = node->i(attr::beg);
          int64_t end_ind = node->i(attr::end);
          return [=](Stack& stack) {
            auto t = pop(stack).toTuple();
            const auto & elems = t->elements();
            std::vector<IValue> output_elems;
            for (int64_t i = beg_ind; i < end_ind; ++i) {
              output_elems.emplace_back(elems.at(i));
            }
            push(stack, Tuple::create(std::move(output_elems)));
            return 0;
          };
        }),
    Operator(
      prim::TupleIndex,
      [](const Node* node) {
        auto index = node->i(attr::index);
        return [=](Stack& stack) {
          auto tup = pop(stack).toTuple();
          const auto & elems = tup->elements();
          // index is normalized to be positive at compile time
          stack.emplace_back(elems.at(index));
          return 0;
        };
      }),
    Operator(
        prim::TupleConstruct,
        [](const Node* node) {
          size_t num_inputs = node->inputs().size();
          return [=](Stack& stack) {
            std::vector<IValue> elems {
              std::make_move_iterator(stack.end() - num_inputs),
              std::make_move_iterator(stack.end())
            };
            drop(stack, num_inputs);
            push(stack, Tuple::create(std::move(elems)));
            return 0;
          };
        }),
    Operator(
        prim::ConstantChunk,
        [](const Node* node) {
          int64_t chunks = node->i(attr::chunks);
          int64_t dim = node->i(attr::dim);
          auto outputs_used = fmap(node->outputs(), [](const Value *v) { return v->uses().size() > 0; });
          return [=](Stack& stack) {
            autograd::profiler::RecordFunction record("chunk");
            at::Tensor t;
            pop(stack, t);
            auto result = at::chunk(t, chunks, dim);
            stack.insert(stack.end(), std::make_move_iterator(result.begin()),
                                      std::make_move_iterator(result.end()));
            // NB: Chunk can sometimes return a smaller number of outputs.
            int64_t num_results = result.size();
            if (num_results != chunks) {
              if (num_results > chunks) {
                JIT_ASSERTM(num_results == chunks,
                            "Expected chunk to return ", chunks, " outputs, but got ", num_results);
              }
              for (int64_t i = num_results; i < chunks; ++i) {
                AT_CHECK(!outputs_used[i],
                         "Expected chunk to return at least ", chunks, " outputs, but got only ", num_results);
                // We know that the output is unused, so it's ok to push anything on the stack.
                stack.emplace_back();
              }
            }
            return 0;
          };
        }),
    Operator(
        prim::ListUnpack,
        [](const Node* node) -> Operation {
          const auto num_outputs = node->outputs().size();
          ListTypePtr lt = node->input()->type()->expect<ListType>();
          if (lt->getElementType() == IntType::get()) {
            return [=](Stack& stack) {
              auto ilist = pop(stack);
              const auto & list = ilist.toIntList()->elements();
              AT_CHECK(list.size() == num_outputs,
                       "Expected ", num_outputs, " elements in a list but found ", list.size());
              stack.insert(stack.end(), list.begin(), list.end());
              return 0;
            };
          } else if (lt->getElementType() == FloatType::get()) {
            return [=](Stack& stack) {
              auto ilist = pop(stack);
              const auto & list = ilist.toDoubleList()->elements();
              AT_CHECK(list.size() == num_outputs,
                       "Expected ", num_outputs, " elements in a list but found ", list.size());
              stack.insert(stack.end(), list.begin(), list.end());
              return 0;
            };
          } else if (lt->getElementType() == DynamicType::get()) {
            return [=](Stack& stack) {
              auto ilist = pop(stack);
              const auto & list = ilist.toTensorList()->elements();
              AT_CHECK(list.size() == num_outputs,
                       "Expected ", num_outputs, " elements in a list but found ", list.size());
              stack.insert(stack.end(), list.begin(), list.end());
              return 0;
            };
          } else {
            AT_ERROR("Unsupported list type: ", lt->getElementType()->str());
          }
        }),
    Operator(
        prim::ListConstruct,
        [](const Node* node) -> Operation {
          const auto num_inputs = node->inputs().size();
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
                vals.emplace_back(std::move(stack[i]).toTensor());
              }
              drop(stack, num_inputs);
              push(stack, std::move(vals));
              return 0;
            };
          } else {
            return [=](Stack& stack) {
              const size_t stack_size = stack.size();
              std::vector<IValue> vals;
              vals.reserve(num_inputs);
              for (size_t i = stack_size - num_inputs; i < stack_size; ++i) {
                vals.emplace_back(std::move(stack[i]));
              }
              drop(stack, num_inputs);
              push(stack, std::move(vals));
              return 0;
            };
          }
        }),
    Operator("aten::_unwrap_optional(t? optional) -> t",
      [](const Node* node) -> Operation {
        return [=](Stack& stack) {
          auto val = pop(stack);
          JIT_ASSERTM(!val.isNone(), "Unwrapping null optional");
          push(stack, val);
          return 0;
        };
      }),
    Operator(
        prim::fork,
        [](const Node* node) {
          Code code(node->g(attr::Subgraph));
          int n_inputs = node->inputs().size();
          JIT_ASSERT(node->blocks().size() == 0);
          JIT_ASSERT(node->hasAttribute(attr::Subgraph));
          return [=](Stack& stack) {
            // Move inputs to a separate stack
            InterpreterState forked_interprester(code);
            InterpreterContinuation continuation(
                forked_interprester,
                Stack(stack.end() - n_inputs, stack.end()));
            drop(stack, n_inputs);

            push(stack, forked_interprester.getFuture());

            c10::global_work_queue.schedule(std::move(continuation));
            return 0;
          };
        }),
    Operator(
        "aten::wait(Future(t) self) -> t",
        [](const Node* node) {
          return [=](Stack& stack) {
            auto future = pop(stack).toFuture();
            if (future->completed()) {
              push(stack, future->value());
            } else {
              throw Suspend(future);
            }
            return 0;
          };
        }),
});

// define implementations for primitive number ops
#define DEFINE_GENERIC_OP(aten_op, int_op, float_op, int_result, float_result) \
  Operator(                                                                    \
      #aten_op "(int a, int b) -> " #int_result,                               \
      [](const Node* node) {                                                         \
        return [=](Stack& stack) {                                             \
          int64_t a, b;                                                        \
          pop(stack, a, b);                                                    \
          push(stack, int_op);                                                 \
          return 0;                                                            \
        };                                                                     \
      }),                                                                      \
  Operator(                                                                    \
      #aten_op "(float a, float b) -> " #float_result, [](const Node* node) {        \
        return [=](Stack& stack) {                                             \
          double a, b;                                                         \
          pop(stack, a, b);                                                    \
          push(stack, float_op);                                               \
          return 0;                                                            \
        };                                                                     \
      }),

#define DEFINE_INT_FLOAT_OP(aten_op, op, result)                               \
  Operator(                                                                    \
      #aten_op "(int a, float b) -> " #result, [](const Node* node) {          \
        return [=](Stack& stack) {                                             \
          int64_t a;                                                           \
          double b;                                                            \
          pop(stack, a, b);                                                    \
          push(stack, op);                                                     \
          return 0;                                                            \
        };                                                                     \
      }),                                                                      \
  Operator(                                                                    \
      #aten_op "(float a, int b) -> " #result, [](const Node* node) {          \
        return [=](Stack& stack) {                                             \
          double a;                                                            \
          int64_t b;                                                           \
          pop(stack, a, b);                                                    \
          push(stack, op);                                                     \
          return 0;                                                            \
        };                                                                     \
      }),


#define DEFINE_INT_OP(aten_op, op)                            \
  Operator(#aten_op "(int a, int b) -> int", [](const Node* node) { \
    return [=](Stack& stack) {                                \
      int64_t a, b;                                           \
      pop(stack, a, b);                                       \
      push(stack, op);                                        \
      return 0;                                               \
    };                                                        \
  }),

#define DEFINE_BINARY_OP(aten_op, op) \
  DEFINE_GENERIC_OP(aten_op, op, op, int, float)  \
  DEFINE_INT_FLOAT_OP(aten_op, op, float)
#define DEFINE_COMPARISON_OP(aten_op, op) \
  DEFINE_GENERIC_OP(aten_op, op, op, bool, bool) \
  DEFINE_INT_FLOAT_OP(aten_op, op, bool)
#define DEFINE_BOOL_OP(aten_op, op)                              \
  Operator(#aten_op "(bool a, bool b) -> bool", [](const Node* node) { \
    return [=](Stack& stack) {                                   \
      bool a, b;                                                 \
      pop(stack, a, b);                                          \
      push(stack, op);                                           \
      return 0;                                                  \
    };                                                           \
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

// Equivalent to list.at(idx)
template <typename TList> // something like Shared<IntList>
typename TList::element_type::ElemType& getItem(TList& list, int64_t idx) {
  const int64_t list_size = list->elements().size();
  const int64_t normalized_idx = normalizeIndex(idx, list_size);
  if (normalized_idx < 0 || normalized_idx >= list_size) {
    throw std::out_of_range("list index out of range");
  }
  return list->elements()[normalized_idx];
}

template <typename TList, typename TElement>
Operation listAppend(const Node* node) {
  return [](Stack& stack) {
    TList a;
    TElement el;
    pop(stack, a, el);

    a->elements().push_back(el);
    push(stack, a);

    return 0;
  };
}

template <typename T>
Operation listSelect(const Node* node) {
  return [=](Stack& stack) {
    T list;
    int64_t idx;
    pop(stack, list, idx);

    auto element = getItem(list, idx);
    push(stack, std::move(element));
    return 0;
  };
}

template <typename T>
Operation listLen(const Node* node) {
  return [=](Stack& stack) {
    T a;
    pop(stack, a);
    const int64_t size = a->elements().size();
    push(stack, size);
    return 0;
  };
}

template <typename T>
Operation listEq(const Node* node) {
  return [=](Stack& stack) {
    T a;
    T b;
    pop(stack, a, b);
    push(stack, a->elements() == b->elements() ? true : false);
    return 0;
  };
}

// Specialization for at::Tensor, since it doesn't define operator==
template <>
Operation listEq<Shared<TensorList>>(const Node* node) {
  return [=](Stack& stack) {
    Shared<TensorList> a;
    Shared<TensorList> b;
    pop(stack, a, b);
    if (a->elements().size() != b->elements().size()) {
      push(stack, false);
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
        push(stack, false);
        return 0;
      }
    }

    push(stack, true);
    return 0;
  };
}

template <class TList, class TElement>
Operation listAdd(const Node* node) {
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
Operation listSlice(const Node* node) {
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

template <typename TList, typename TElement>
Operation listSetItem(const Node* node) {
  return [](Stack& stack) {
    TList list;
    int64_t idx;
    TElement value;

    pop(stack, list, idx, value);
    getItem(list, idx) = value;

    push(stack, list);
    return 0;
  };
}

RegisterOperators reg2({

#define DEFINE_STRING_OP(op_name, string_op, result)                           \
Operator(                                                                      \
    #op_name "(str a, str b) ->" #result,                                \
    [](const Node* node) {                                                    \
      return [=](Stack& stack) {                                               \
        auto b = pop(stack).toStringRef();                                     \
        auto a = pop(stack).toStringRef();                                     \
        push(stack, string_op);                                                \
        return 0;                                                              \
    };                                                                         \
  }),

  DEFINE_STRING_OP(aten::eq, a == b, bool)
  DEFINE_STRING_OP(aten::ne, a != b, bool)
  DEFINE_STRING_OP(aten::add, a + b, str)
#undef DEFINE_STRING_OP

    // tensor length op (size of 1st dimension)
    Operator(
      "aten::len(Tensor t) -> int",
      [](Stack& stack) {
        at::Tensor t = pop(stack).toTensor();
        if (t.dim() == 0) {
          AT_ERROR("len() of a 0-d tensor");
        }
        push(stack, t.sizes()[0]);
        return 0;
      }
    ),
#define CREATE_LIST_OPS(decl_type, c_type) \
    Operator("aten::select(" decl_type "[] a, int b) -> " decl_type, listSelect<Shared<c_type>>), \
    Operator("aten::_set_item(" decl_type "[](a!) l, int idx, " decl_type " el) -> " decl_type"[](a!)", listSetItem<Shared<c_type>, c_type::ElemType>), \
    Operator("aten::len(" decl_type "[] a) -> int", listLen<Shared<c_type>>), \
    Operator("aten::add(" decl_type "[] a, " decl_type "[] b) -> " decl_type "[]", listAdd<Shared<c_type>, c_type::ElemType>), \
    Operator( \
        "aten::slice(" decl_type "[] l, int start, int end=9223372036854775807, int step=1) -> " decl_type "[]", \
        listSlice<Shared<c_type>, c_type::ElemType>), \
    Operator( \
        "aten::append(" decl_type "[](a!) self, " decl_type " el) -> " decl_type "[](a!)", \
        listAppend<Shared<c_type>, c_type::ElemType>), \


    CREATE_LIST_OPS("int", IntList)
    CREATE_LIST_OPS("float", DoubleList)
    CREATE_LIST_OPS("Tensor", TensorList)
    CREATE_LIST_OPS("t", GenericList)
#undef CREATE_LIST_OPS


    Operator("aten::eq(int[] a, int[] b) -> bool", listEq<Shared<IntList>>),
    Operator("aten::eq(float[] a, float[] b) -> bool", listEq<Shared<DoubleList>>),
    Operator("aten::eq(Tensor[] a, Tensor[] b) -> bool", listEq<Shared<TensorList>>),

#define CREATE_COPY_OP(other_type, c_type)                              \
  Operator(                                                             \
      "aten::copy_(Tensor(a!) t, " #other_type " other) -> Tensor(a!)", \
      [](const Node* node) {                                            \
        return [=](Stack& stack) {                                      \
          at::Tensor t;                                                 \
          c_type other;                                                 \
          pop(stack, t, other);                                         \
          std::move(t) = other;                                         \
          push(stack, std::move(t));                                    \
          return 0;                                                     \
        };                                                              \
      }),

    CREATE_COPY_OP(Tensor, at::Tensor)
    CREATE_COPY_OP(int, int64_t)
    CREATE_COPY_OP(float, double)
#undef CREATE_COPY_OP

    DEFINE_BINARY_OP(aten::add, a + b)
    DEFINE_BINARY_OP(aten::sub, a - b)
    DEFINE_BINARY_OP(aten::mul, a * b)
    DEFINE_BINARY_OP(aten::pow, static_cast<decltype(a)>(pow(a, b)))

    // Pass in two ops for handling int and float separately as % in C++ only works for int
    // The modulus calculation is different between C++ and Python (on negative), we preserve
    // the python behavior as it's more common and match python syntax, hence the conversion.
    DEFINE_GENERIC_OP(aten::remainder, (b + (a % b)) % b, fmod((b + fmod(a, b)), b), int, float)
    DEFINE_INT_FLOAT_OP(aten::remainder, fmod((b + fmod(a, b)), b), float)


    // in c++ int division rounds to the integer closer to 0, in python floordiv
    // rounds to lower integer
    DEFINE_GENERIC_OP(aten::floordiv,
      static_cast<int64_t>(std::floor(static_cast<double>(a) / static_cast<double>(b))),
      std::floor(a / b), int, float)
    DEFINE_INT_FLOAT_OP(aten::floordiv, std::floor(a / b), float)

    //only used in loop unrolling, not exposed to end users
    DEFINE_INT_OP(aten::__round_to_zero_floordiv, a / b)

    DEFINE_INT_OP(aten::__and__, a & b)
    DEFINE_INT_OP(aten::__or__, a | b)
    DEFINE_INT_OP(aten::__xor__, a ^ b)

    // NB: This is the python truediv operation
    Operator("aten::div(int a, int b) -> float",
        [](const Node* node) {
          return [=](Stack& stack) {
            int64_t a, b;
            pop(stack, a, b);
            push(stack, static_cast<double>(a) / static_cast<double>(b));
            return 0;
          };
        }),
    Operator("aten::div(float a, float b) -> float",
        [](const Node* node) {
          return [=](Stack& stack) {
            double a, b;
            pop(stack, a, b);
            push(stack, a / b);
            return 0;
          };
        }),

    DEFINE_COMPARISON_OP(aten::ne, a != b)
    DEFINE_COMPARISON_OP(aten::eq, a == b)
    DEFINE_COMPARISON_OP(aten::lt, a < b)
    DEFINE_COMPARISON_OP(aten::gt, a > b)
    DEFINE_COMPARISON_OP(aten::le, a <= b)
    DEFINE_COMPARISON_OP(aten::ge, a >= b)

    DEFINE_BOOL_OP(aten::__and__, a && b)
    DEFINE_BOOL_OP(aten::__or__, a || b)
    DEFINE_BOOL_OP(aten::__xor__, a != b)

    Operator(
        "aten::neg(int self) -> int",
        [](const Node* node) {
          return [=](Stack& stack) {
            push(stack, -pop(stack).toInt());
            return 0;
          };
        }),
    Operator(
        "aten::neg(float self) -> float",
        [](const Node* node) {
          return [=](Stack& stack) {
            push(stack, -pop(stack).toDouble());
            return 0;
          };
        }),
    Operator(
        "aten::__not__(bool self) -> bool",
        [](const Node* node) {
          return [=](Stack& stack) {
            push(stack, !pop(stack).toBool());
            return 0;
          };
        }),
    Operator(
        "aten::__is__(t1 self, t2 obj) -> bool",
        [](const Node* node) {
          return [=](Stack& stack) {
            IValue self, obj;
            pop(stack, self, obj);
            push(stack, self.isSameIdentity(obj));
            return 0;
          };
        }),
    Operator(
        "aten::__isnot__(t1 self, t2 obj) -> bool",
        [](const Node* node) {
          return [=](Stack& stack) {
            IValue self, obj;
            pop(stack, self, obj);
            push(stack, !self.isSameIdentity(obj));
            return 0;
          };
        }),
    Operator(
        "aten::_tensor_to_list(Tensor self) -> int[]",
        [](const Node* node) {
          return [=](Stack& stack) {
            at::Tensor t;
            pop(stack, t);
            std::vector<int64_t> elems;
            for(int i = 0; i < t.size(0); i++){
              elems.push_back(*t[i].data<int32_t>());
            }
            push(stack, jit::IntList::create(elems));
            return 0;
          };
        }),
    Operator(
        "aten::_list_to_tensor(int[] self) -> Tensor",
        [](const Node* node) {
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
});


at::Tensor leaky_relu(at::Tensor tensor, double scalar) {
  return at::leaky_relu(tensor, scalar);
}
at::Tensor cat(std::vector<at::Tensor> tensors) {
  return at::cat(tensors);
}

static auto reg3 =
    torch::jit::RegisterOperators()
        .op("_test::leaky_relu(Tensor self, float v=0.01) -> Tensor", &leaky_relu)
        .op("_test::cat(Tensor[] inputs) -> Tensor", &cat);

}}} // torch::jit::anon
