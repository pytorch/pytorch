#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/fusers/interface.h"
#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/custom_operator.h"

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

Operation noop(Node* n) {
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
        prim::MemoryFence,
        [](Node* node) {
          return [](Stack& stack) {
            return 0;
          };
        }),
    Operator(
        prim::FusionGroup,
        [](Node* node) {
          auto handle = getFusionHandle(node);
          return [handle](Stack& stack) {
            autograd::profiler::RecordFunction record("FusionGroup");
            handle->run(stack);
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
              push(stack, a.item<int64_t>());
              return 0;
            };
          } else {
            return [](Stack& stack) {
              at::Tensor a;
              pop(stack, a);
              at::DeviceGuard guard(a);
              push(stack, a.item<double>());
              return 0;
            };
          }
        }),
    Operator(
        prim::ImplicitTensorToNum,
        [](Node* node) -> Operation {
          if(node->output()->type() == IntType::get()) {
            return [](Stack& stack) {
              at::Tensor a;
              pop(stack, a);
              checkImplicitTensorToNum(a, /*to int*/true);
              at::DeviceGuard guard(a);
              push(stack, a.item<int64_t>());
              return 0;
            };
          } else {
            return [](Stack& stack) {
              at::Tensor a;
              pop(stack, a);
              checkImplicitTensorToNum(a, /*to int*/false);
              at::DeviceGuard guard(a);
              push(stack, a.item<double>());
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
            push(stack, autograd::make_variable(at::scalar_to_tensor(s)));
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
        prim::StringToFloat,
        [](Node* node) -> Operation {
          return [](Stack& stack) {
            auto s = pop(stack).toString();
            if (s->string() != "inf") {
              AT_ERROR(
                  "Only 'inf' can be cast to a float, but got '",
                  s->string(),
                  "'");
            }
            push(stack, std::numeric_limits<double>::infinity());
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
        prim::NoneGenerator,
        [](Node* node) {
          return [](Stack& stack) {
            stack.emplace_back();
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
        prim::LoadWorld,
        [](Node* node) {
          return [](Stack& stack) {
            push(stack, World());
            return 0;
          };
        }),
    Operator(
        prim::DummyWorld,
        [](Node* node) {
          return [](Stack& stack) {
            AT_ERROR("Encountered a dummy world during graph execution.");
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
        prim::TupleUnpack,
        [](Node* node) {
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
        prim::TupleConstruct,
        [](Node* node) {
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
        [](Node* node) {
          int64_t chunks = node->i(attr::chunks);
          int64_t dim = node->i(attr::dim);
          auto outputs_used = fmap(node->outputs(), [](Value *v) { return v->uses().size() > 0; });
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
        [](Node* node) -> Operation {
          size_t num_outputs = node->outputs().size();
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
            return [=](Stack& stack) {
              const size_t stack_size = stack.size();
              std::vector<IValue> vals;
              vals.reserve(num_inputs);
              for (size_t i = stack_size - num_inputs; i < stack_size; ++i) {
                vals.push_back(std::move(stack[i]));
              }
              drop(stack, num_inputs);
              push(stack, std::move(vals));
              return 0;
            };
          }
        }),
});

// define implementations for primitive number ops
#define DEFINE_GENERIC_OP(aten_op, int_op, float_op, float_result)          \
  Operator(                                                                 \
      #aten_op "(int a, int b) -> int",                                     \
      [](Node* node) {                                                      \
        return [=](Stack& stack) {                                          \
          int64_t a, b;                                                     \
          pop(stack, a, b);                                                 \
          push(stack, int_op);                                                  \
          return 0;                                                         \
        };                                                                  \
      }),                                                                   \
  Operator(                                                                 \
      #aten_op "(float a, float b) -> " #float_result, [](Node* node) {     \
        return [=](Stack& stack) {                                          \
          double a, b;                                                      \
          pop(stack, a, b);                                                 \
          push(stack, float_op);                                                  \
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

#define DEFINE_BINARY_OP(aten_op, op) DEFINE_GENERIC_OP(aten_op, op, op, float)
#define DEFINE_COMPARISON_OP(aten_op, op) DEFINE_GENERIC_OP(aten_op, op, op, int)

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

template <typename TList, typename TElement>
Operation listAppend(Node* node) {
  return [](Stack& stack) {
    TList a;
    TElement el;
    pop(stack, a, el);

    a->elements().push_back(el);

    return 0;
  };
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
    push(stack, a->elements() == b->elements() ? 1 : 0);
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

#define CREATE_LIST_OPS(decl_type, c_type) \
    // Select element in the `b`th position from list `a`
    // Equivalent to `a[b]` in Python.
    Operator("aten::select(" decl_type "[] a, int b) -> " decl_type, listSelect<Shared<c_type>>), \
    // Return the size of list `a`
    // Equivalent to `len(a)` in Python.
    Operator("aten::len(" decl_type "[] a) -> int", listLen<Shared<c_type>>), \
    Operator("aten::add(" decl_type "[] a, " decl_type "[] b) -> " decl_type "[]", listAdd<Shared<c_type>, c_type::ElemType>), \
    // Return a slice of list `l`, with a specified start, end, and step length
    // Equivalent to `l[start:end:step]` in Python.
    Operator( \
        "aten::slice(" decl_type "[] l, int start, int end=9223372036854775807, int step=1) -> " decl_type "[]", \
        listSlice<Shared<c_type>, c_type::ElemType>),
    // Append `el` to `list`
    // Equivalent to `list.append(el)` in Python.
    Operator( \
        "aten::append(World w, " decl_type "[] list, " decl_type " el) -> World", \
        listAppend<Shared<c_type>, c_type::ElemType>),


    CREATE_LIST_OPS("int", IntList)
    CREATE_LIST_OPS("float", DoubleList)
    CREATE_LIST_OPS("Tensor", TensorList)
    CREATE_LIST_OPS("t", GenericList)


    Operator("aten::eq(int[] a, int[] b) -> int", listEq<Shared<IntList>>),
    Operator("aten::eq(float[] a, float[] b) -> int", listEq<Shared<DoubleList>>),
    Operator("aten::eq(Tensor[] a, Tensor[] b) -> int", listEq<Shared<TensorList>>),

    DEFINE_BINARY_OP(aten::add, a + b)
    DEFINE_BINARY_OP(aten::sub, a - b)
    DEFINE_BINARY_OP(aten::mul, a * b)
    DEFINE_BINARY_OP(aten::pow, static_cast<decltype(a)>(pow(a, b)))

    // Pass in two ops for handling int and float separately as % in C++ only works for int
    // The modulus calculation is different between C++ and Python (on negative), we preserve
    // the python behavior as it's more common and match python syntax, hence the conversion.
    DEFINE_GENERIC_OP(aten::remainder, (b + (a % b)) % b, fmod((b + fmod(a, b)), b), float)

    // TODO: Support python floordiv (//)
    // Right now aten::floordiv is only used by loop unrolling
    DEFINE_INT_OP(aten::floordiv, a / b)

    // NB: This is the python truediv operation
    Operator("aten::div(int a, int b) -> float",
        [](Node* node) {
          return [=](Stack& stack) {
            int64_t a, b;
            pop(stack, a, b);
            push(stack, static_cast<double>(a) / static_cast<double>(b));
            return 0;
          };
        }),
    Operator("aten::div(float a, float b) -> float",
        [](Node* node) {
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
              elems.push_back(*t[i].data<int32_t>());
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
