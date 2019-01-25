#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/api/include/torch/utils.h>
#include <aten/src/ATen/ExpandUtils.h>

#include <c10/core/ScalarType.h>
#include <aten/src/ATen/InitialTensorOptions.h>
#include <torch/csrc/jit/script/error_report.h>

#include <regex>
#include <sstream>

namespace torch {
namespace jit {

namespace {


void checkListInputType(const c10::TypePtr& elem_type, const Node* node) {
  if (!elem_type->isSubtypeOf(NumberType::get()) && elem_type != BoolType::get()) {
    auto error = script::ErrorReport(node->getSourceLocation());
    error << "Input list to torch.tensor must be of ints, floats, or bools, " <<
      "got " << elem_type->str();
    // special case empty list torch.tensor([])
    if (elem_type->isSubtypeOf(DynamicType::get())) {
      auto input = node->inputs().at(0);
      if (input->node()->kind() == prim::ListConstruct && input->node()->inputs().size() == 0) {
        error << "\n(Note: empty lists are constructed as Tensor[]; \n"
               << "if you want an empty list of a different type, \n"
               << "use `torch.jit.annotate(List[T], [])`, \n"
               << "where `T` is the type of elements in the list)";
      }
    }
    throw error;
  }
}

at::ScalarType scalarTypeFromJitType(const c10::TypePtr& type) {
  if (type == FloatType::get()) {
    return at::ScalarType::Double;
  } else if (type == IntType::get()) {
    return at::ScalarType::Long;
  } else if (type == BoolType::get()) {
    return at::ScalarType::Byte;
  }
  AT_ASSERTM(0, "Add new condition, expected Float, Int, or Bool but got",
      type->str());
}


int64_t list_size(const IValue& list) {
  if (list.isGenericList()) {
    return list.toGenericListRef().size();
  } else if (list.isIntList()) {
    return list.toIntListRef().size();
  } else if (list.isDoubleList()){
    return list.toDoubleListRef().size();
  } else if (list.isBoolList()) {
    return list.toBoolListRef().size();
  }
  AT_ASSERTM(0, "Unexpected list type", list);
}

std::vector<int64_t> compute_sizes(const IValue& seq) {
  std::vector<int64_t> sizes;
  // because bool, int, and float lists are specialized, inner array will
  // will not be generic list
  auto seq_recur = seq;
  while (seq_recur.isGenericList()) {
    auto seq_list = seq_recur.toGenericListRef();
    auto length = seq_list.size();
    AT_ASSERT(length != 0);
    sizes.push_back(length);
    seq_recur = seq_list[0];
  }
  sizes.push_back(list_size(seq_recur));
  return sizes;
}

void checkSequenceSize(int64_t n, int64_t dim, int64_t seq_size) {
  if (seq_size != n) {
    AT_ERROR("Expected sequence of length ", n, " at dim ", dim, " (got ", seq_size, ")");
  }
}

template <typename DTYPE>
void storeLastDimension(char* data, const std::vector<int64_t>& sizes, const c10::ArrayRef<int64_t>& strides, int64_t dim,
    int elementSize, const std::vector<DTYPE>& obj) {
  auto n = sizes[dim];
  auto seq_size = obj.size();
  checkSequenceSize(n, dim, seq_size);
  for (int64_t i = 0; i < n; i++) {
    *(DTYPE*)data = obj[i];
    data += strides[dim] * elementSize;
  }
}

// bool vector needs to be cast to uint8_t
template<>
void storeLastDimension<bool>(char* data, const std::vector<int64_t>& sizes, const c10::ArrayRef<int64_t>& strides, int64_t dim,
    int elementSize, const std::vector<bool>& obj) {
  auto n = sizes[dim];
  auto seq_size = obj.size();
  checkSequenceSize(n, dim, seq_size);
  for (int64_t i = 0; i < n; i++) {
    *(uint8_t*)data = static_cast<uint8_t>(obj[i]);
    data += strides[dim] * elementSize;
  }
}

// refernce python implementation recursive_store in tensor_new.cpp

void recursiveStore(char* data, const std::vector<int64_t>& sizes, const c10::ArrayRef<int64_t>& strides, int64_t dim,
   int elementSize, const IValue& obj) {

  auto ndim = sizes.size();
  auto n = sizes[dim];
  auto seq_size = list_size(obj);
  checkSequenceSize(n, dim, seq_size);
  if (dim + 1 < static_cast<long>(ndim)) {
    auto items = obj.toGenericListRef();
    for (int64_t i = 0; i < n; i++) {
      recursiveStore(data, sizes, strides, dim + 1, elementSize, items[i]);
      data += strides[dim] * elementSize;
    }
  } else {
    JIT_ASSERT(obj.isIntList() || obj.isDoubleList() || obj.isBoolList());
    if (obj.isIntList()) {
      storeLastDimension<int64_t>(data, sizes, strides, dim, elementSize, obj.toIntListRef());
    } else if (obj.isDoubleList()){
      storeLastDimension<double>(data, sizes, strides, dim, elementSize, obj.toDoubleListRef());
    } else {
      storeLastDimension<bool>(data, sizes, strides, dim, elementSize, obj.toBoolListRef());
    }
  }
}

RegisterOperators reg({
    Operator(
        "aten::split(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]",
        [](Stack& stack) {
          autograd::profiler::RecordFunction record("split_with_sizes");
          auto result = at::split_with_sizes(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements(),
              (std::move(peek(stack, 2, 3))).toInt());
          drop(stack, 3);
          pack(stack, std::move(result));
          return 0;
        }),
    Operator(
        "aten::Size(int[] sizes) -> int[]",
        [](Stack& stack) { return 0; }),
    Operator(
        "aten::size(Tensor self) -> int[]",
        [](Stack& stack) {
          autograd::profiler::RecordFunction record("sizes");
          auto t = std::move(pop(stack)).toTensor();
          pack(stack, t.sizes().vec());
          return 0;
        }),
    Operator(
        "aten::list_with_default(int[] list, int[] defaults) -> int[]",
        [](Stack& stack) {
          autograd::profiler::RecordFunction record("sizes");
          auto list = peek(stack, 0, 2).toIntListRef();
          auto defaults = peek(stack, 1, 2).toIntListRef();
          drop(stack, 2);

          JIT_ASSERT(defaults.size() > list.size());

          // TODO: allow list of optionals to be filled in with defaults
          // i.e. list_with_default([1, 2, None], [1, 2, 3]) -> [1, 2, 3]

          push(stack, list);
          return 0;
        }),
    Operator(
        "aten::_infer_size(int[] a, int[] b) -> int[]",
        [](const Node* node) {
          return [](Stack& stack) {
            auto a = pop(stack).toIntList()->elements();
            auto b = pop(stack).toIntList()->elements();
            push(stack, at::infer_size(a, b));
            return 0;
          };
        }),
    Operator(
      "aten::_no_grad_embedding_renorm_(Tensor weight, Tensor input, float max_norm, float norm_type) -> Tensor",
      [](const Node* node) {
        return [](Stack& stack) {
          at::Tensor weight;
          at::Tensor input;
          double max_norm;
          double norm_type;
          pop(stack, weight, input, max_norm, norm_type);

          // TODO: remove when script supports setting grad mode
          torch::NoGradGuard no_grad;

          at::Tensor result = at::embedding_renorm_(weight, input, max_norm, norm_type);
          push(stack, result);

          return 0;
        };
      }),
    Operator(
        "aten::format(str self, ...) -> str",
        [](const Node* node) {
          size_t num_inputs = node->inputs().size();
          std::regex unsupported_options("\\{(.*)\\}");
          return [num_inputs, unsupported_options](Stack& stack) {
            auto format = peek(stack, 0, num_inputs).toStringRef();

            if (std::regex_search(format, unsupported_options)) {
              AT_WARN("Format options are not supported.");
            }

            auto args = last(stack, num_inputs - 1);
            std::stringstream ss;
            for (size_t begin = 0, used_args = 0; true; ++used_args) {
              size_t loc = format.find("{}", begin);
              if (loc == std::string::npos) {
                ss << format.substr(begin);
                break;
              }
              ss << format.substr(begin, loc - begin);
              if (used_args >= args.size()) {
                AT_ERROR("Too few arguments for format string: ", format);
              }
              ss << args[used_args];
              begin = loc + 2;
            }

            drop(stack, num_inputs);
            push(stack, ss.str());
            return 0;
          };
        }),

#define DEFINE_TORCH_TENSOR_OP(operator_type, c_type, tensor_creation_op)             \
Operator(                                                                             \
  "aten::tensor(" #operator_type " t, *, ScalarType? dtype=None, Device? device=None"\
      ") -> Tensor",                                                                  \
  [](const Node* node) {                                                              \
    auto initial_scalar_type = scalarTypeFromJitType(node->inputs().at(0)->type());   \
    return [initial_scalar_type](Stack& stack) {                                      \
      c_type scalar_val;                                                              \
      IValue dtype;                                                                   \
      IValue device;                                                                  \
      pop(stack, scalar_val, dtype, device);                                          \
      auto tensor = autograd::make_variable(tensor_creation_op);                      \
      at::ScalarType scalar_type = dtype.isNone() ?                                   \
        tensor.scalar_type() : dtype.toScalarType();                                  \
      c10::Device dev = device.isNone() ? tensor.device() : device.toDevice();        \
      if (scalar_type != initial_scalar_type || dev != tensor.device()) {             \
        tensor = tensor.to(dev, scalar_type);                                         \
      }                                                                               \
      push(stack, tensor);                                                            \
      return 0;                                                                       \
    };                                                                                \
  }),

DEFINE_TORCH_TENSOR_OP(float, double, at::scalar_to_tensor(scalar_val))
DEFINE_TORCH_TENSOR_OP(int, int64_t, at::scalar_to_tensor(scalar_val))
DEFINE_TORCH_TENSOR_OP(bool, bool, at::empty({}, at::CPU(at::kByte).options()).fill_(scalar_val))


    // reference python implementation: internal_new_from_data in tensor_new.cpp
    Operator(
        "aten::_infer_size(int[] a, int[] b) -> int[]",
        [](const Node* node) {
          return [](Stack& stack) {
            auto a = pop(stack).toIntList()->elements();
            auto b = pop(stack).toIntList()->elements();
            push(stack, at::infer_size(a, b));
            return 0;
          };
        }),
    Operator(
        "aten::_no_grad_embedding_renorm_(Tensor weight, Tensor input, float max_norm, float norm_type) -> Tensor",
        [](const Node* node) {
          return [](Stack& stack) {
            at::Tensor weight;
            at::Tensor input;
            double max_norm;
            double norm_type;
            pop(stack, weight, input, max_norm, norm_type);

            // TODO: remove when script supports setting grad mode
            torch::NoGradGuard no_grad;

            at::Tensor result =
                at::embedding_renorm_(weight, input, max_norm, norm_type);
            push(stack, result);

            return 0;
          };
        }),
    Operator(
      "aten::tensor(t[] data, *, ScalarType? dtype=None, Device? device=None) -> Tensor",
      [](const Node* node) {
        auto input = node->inputs().at(0);
        auto elem_type = input->type();
        while (auto list_type = elem_type->cast<ListType>()) {
          elem_type = list_type->getElementType();
        }
        checkListInputType(elem_type, node);
        at::ScalarType initial_scalar_type = scalarTypeFromJitType(elem_type);
        return [initial_scalar_type, elem_type](Stack& stack) {
          IValue data;
          IValue dtype;
          IValue device;
          pop(stack, data, dtype, device);
          auto sizes = compute_sizes(data);
          auto tensor = autograd::make_variable(
            at::empty(sizes, at::initialTensorOptions().dtype(initial_scalar_type)));

          recursiveStore((char*)tensor.data_ptr(), sizes, tensor.strides(), 0,
              tensor.type().elementSizeInBytes(), data);

          at::ScalarType scalar_type = dtype.isNone() ? tensor.scalar_type() : dtype.toScalarType();
          c10::Device dev = device.isNone() ? tensor.device() : device.toDevice();
          if (scalar_type != initial_scalar_type || dev != tensor.device()) {
            tensor = tensor.to(dev, scalar_type);
          }

          auto default_type = at::typeMetaToScalarType(at::get_default_dtype());

          if (dtype.isNone() && tensor.scalar_type() != default_type &&
              tensor.numel() == 0) {
            AT_WARN("Creating a tensor from an empty ", elem_type->str(),
              "list will create a tensor of default floating point type  (currently ", default_type,
              ") in python but a tensor of type ", elem_type->str(), " in torchscript.\n",
              "Pass in a dtype argument to ensure consistent behavior");
          }

          push(stack, tensor);
          return 0;
        };
      }),
    Operator(
        "aten::_assert_int_or_pair(int[] vals, str name, str message) -> Tensor",
        [](const Node* node) {
          return [](Stack& stack) {
            // Everything is a list at the point this is used, so don't do
            // anything
            drop(stack, 3);
            return 0;
          };
        }),

});
}

// define implementations for primitive number ops
#define DEFINE_GENERIC_OP(aten_op, int_op, float_op, int_result, float_result) \
  Operator(                                                                    \
      #aten_op "(int a, int b) -> " #int_result,                               \
      [](const Node* node) {                                                   \
        return [=](Stack& stack) {                                             \
          int64_t a, b;                                                        \
          pop(stack, a, b);                                                    \
          push(stack, int_op);                                                 \
          return 0;                                                            \
        };                                                                     \
      }),                                                                      \
      Operator(                                                                \
          #aten_op "(float a, float b) -> " #float_result,                     \
          [](const Node* node) {                                               \
            return [=](Stack& stack) {                                         \
              double a, b;                                                     \
              pop(stack, a, b);                                                \
              push(stack, float_op);                                           \
              return 0;                                                        \
            };                                                                 \
          })

#define DEFINE_INT_FLOAT_OP(aten_op, op, result)                               \
  Operator(                                                                    \
      #aten_op "(int a, float b) -> " #result,                                 \
      [](const Node* node) {                                                   \
        return [=](Stack& stack) {                                             \
          int64_t a;                                                           \
          double b;                                                            \
          pop(stack, a, b);                                                    \
          push(stack, op);                                                     \
          return 0;                                                            \
        };                                                                     \
      }),                                                                      \
      Operator(#aten_op "(float a, int b) -> " #result, [](const Node* node) { \
        return [=](Stack& stack) {                                             \
          double a;                                                            \
          int64_t b;                                                           \
          pop(stack, a, b);                                                    \
          push(stack, op);                                                     \
          return 0;                                                            \
        };                                                                     \
      })

#define DEFINE_INT_OP(aten_op, op)                                  \
  Operator(#aten_op "(int a, int b) -> int", [](const Node* node) { \
    return [=](Stack& stack) {                                      \
      int64_t a, b;                                                 \
      pop(stack, a, b);                                             \
      push(stack, op); /* NOLINT(hicpp-signed-bitwise) */           \
      return 0;                                                     \
    };                                                              \
  })

#define DEFINE_BINARY_OP(aten_op, op)             \
  DEFINE_GENERIC_OP(aten_op, op, op, int, float), \
      DEFINE_INT_FLOAT_OP(aten_op, op, float)
#define DEFINE_COMPARISON_OP(aten_op, op)         \
  DEFINE_GENERIC_OP(aten_op, op, op, bool, bool), \
      DEFINE_INT_FLOAT_OP(aten_op, op, bool)
#define DEFINE_BOOL_OP(aten_op, op)                                    \
  Operator(#aten_op "(bool a, bool b) -> bool", [](const Node* node) { \
    return [=](Stack& stack) {                                         \
      bool a, b;                                                       \
      pop(stack, a, b);                                                \
      push(stack, op);                                                 \
      return 0;                                                        \
    };                                                                 \
  })

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

// cannot return a reference to an element in a bool vector
bool getBoolItem(const std::vector<bool>& list, int64_t idx) {
  const int64_t list_size = list.size();
  const int64_t normalized_idx = normalizeIndex(idx, list_size);
  if (normalized_idx < 0 || normalized_idx >= list_size) {
    throw std::out_of_range("list index out of range");
  }
  return list[normalized_idx];
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

// needs specialization because cannot return a pointer to a bool in an array
template<>
Operation listSelect<Shared<BoolList>>(const Node* node) {
  return [=](Stack& stack) {
    Shared<BoolList> list;
    int64_t idx;
    pop(stack, list, idx);

    auto element = getBoolItem(list->elements(), idx);
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

template <typename T>
Operation listNe(const Node* node) {
  return [=](Stack& stack) {
    T a;
    T b;
    pop(stack, a, b);
    push(stack, !(a->elements() == b->elements()));
    return 0;
  };
}

inline bool tensor_list_equal(Shared<TensorList> a, Shared<TensorList> b) {
  if (a->elements().size() != b->elements().size()) {
    return false;
  }

  for (size_t i = 0; i < a->elements().size(); ++i) {
    const auto& a_element = a->elements()[i];
    const auto& b_element = b->elements()[i];
    // This preserves Python's semantics, which uses eq() to compare two
    // elements, then passes the result to bool().
    // see: https://docs.python.org/3.4/reference/datamodel.html#object.__ge__
    const auto cmp_result = a_element.eq(b_element);
    if (!cmp_result.is_nonzero()) {
      return false;
    }
  }

  return true;
}

// Specialization for at::Tensor, since it doesn't define operator==
template <>
Operation listEq<Shared<TensorList>>(const Node* node) {
  return [=](Stack& stack) {
    Shared<TensorList> a;
    Shared<TensorList> b;
    pop(stack, a, b);
    push(stack, tensor_list_equal(a, b));
    return 0;
  };
}

// Specialization for at::Tensor, since it doesn't define operator==
template <>
Operation listNe<Shared<TensorList>>(const Node* node) {
  return [=](Stack& stack) {
    Shared<TensorList> a;
    Shared<TensorList> b;
    pop(stack, a, b);
    push(stack, !tensor_list_equal(a, b));
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


template<>
Operation listSetItem<Shared<BoolList>, bool>(const Node* node) {
  return [](Stack& stack) {
    Shared<BoolList> list;
    int64_t idx;
    bool value;

    pop(stack, list, idx, value);

    int64_t list_size = list->elements().size();
    auto normalized_idx = normalizeIndex(idx, list_size);
    if (normalized_idx < 0 || normalized_idx >= list_size) {
      throw std::out_of_range("list index out of range");
    }
    list->elements()[normalized_idx] = value;

    push(stack, list);
    return 0;
  };
}


RegisterOperators reg2({

  #define DEFINE_STRING_OP(op_name, string_op, result)                    \
    Operator(#op_name "(str a, str b) ->" #result, [](const Node* node) { \
      return [=](Stack& stack) {                                          \
        auto b = pop(stack).toStringRef();                                \
        auto a = pop(stack).toStringRef();                                \
        push(stack, string_op);                                           \
        return 0;                                                         \
      };                                                                  \
    })

      DEFINE_STRING_OP(aten::eq, a == b, bool),
      DEFINE_STRING_OP(aten::ne, a != b, bool),
      DEFINE_STRING_OP(aten::add, a + b, str),
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
          }),
      Operator(
          "aten::append(Tensor[](a!) self, Tensor(c) el) -> Tensor[](a!)",
          listAppend<Shared<TensorList>, at::Tensor>),
      Operator(
          "aten::select(Tensor[](a) list, int idx) -> Tensor(*)",
          listSelect<Shared<TensorList>>),
      Operator(
          "aten::_set_item(Tensor[](a!) l, int idx, Tensor el) -> Tensor[](a!)",
          listSetItem<Shared<TensorList>, at::Tensor>),

  // Mutable ops for lists containing immutable types.
  #define CREATE_IMMUTABLE_LIST_OPS(decl_type, c_type)                   \
    Operator(                                                            \
        "aten::select(" decl_type "[] a, int b) -> " decl_type,          \
        listSelect<Shared<c_type>>),                                     \
        Operator(                                                        \
            "aten::append(" decl_type "[](a!) self, " decl_type          \
            " el) -> " decl_type "[](a!)",                               \
            listAppend<Shared<c_type>, c_type::ElemType>),               \
        Operator(                                                        \
            "aten::_set_item(" decl_type "[](a!) l, int idx, " decl_type \
            " el) -> " decl_type "[](a!)",                               \
            listSetItem<Shared<c_type>, c_type::ElemType>)

      CREATE_IMMUTABLE_LIST_OPS("int", IntList),
      CREATE_IMMUTABLE_LIST_OPS("float", DoubleList),
      CREATE_IMMUTABLE_LIST_OPS("t", GenericList),
      CREATE_IMMUTABLE_LIST_OPS("bool", BoolList),

  #define CREATE_LIST_OPS(decl_type, c_type)                                          \
    Operator("aten::len(" decl_type "[] a) -> int", listLen<Shared<c_type>>),         \
        Operator(                                                                     \
            "aten::add(" decl_type "[] a, " decl_type "[] b) -> " decl_type           \
            "[]",                                                                     \
            listAdd<Shared<c_type>, c_type::ElemType>),                               \
        Operator(                                                                     \
            "aten::slice(" decl_type                                                  \
            "[] l, int start, int end=9223372036854775807, int step=1) -> " decl_type \
            "[]",                                                                     \
            listSlice<Shared<c_type>, c_type::ElemType>)

      CREATE_LIST_OPS("int", IntList),
      CREATE_LIST_OPS("float", DoubleList),
      CREATE_LIST_OPS("Tensor", TensorList),
      CREATE_LIST_OPS("t", GenericList),
  #undef CREATE_LIST_OPS

      Operator("aten::eq(int[] a, int[] b) -> bool", listEq<Shared<IntList>>),
      Operator(
          "aten::eq(float[] a, float[] b) -> bool",
          listEq<Shared<DoubleList>>),
      Operator(
          "aten::eq(Tensor[] a, Tensor[] b) -> bool",
          listEq<Shared<TensorList>>),
      Operator(
          "aten::eq(bool[] a, bool[] b) -> bool",
          listEq<Shared<BoolList>>),
      Operator("aten::ne(int[] a, int[] b) -> bool", listNe<Shared<IntList>>),
      Operator(
          "aten::ne(float[] a, float[] b) -> bool",
          listNe<Shared<DoubleList>>),
      Operator(
          "aten::ne(Tensor[] a, Tensor[] b) -> bool",
          listNe<Shared<TensorList>>),
      Operator(
          "aten::ne(bool[] a, bool[] b) -> bool",
          listNe<Shared<BoolList>>),


  #define CREATE_COPY_OP(other_type, c_type)                                 \
    Operator(                                                                \
        "aten::copy_(Tensor(a!) self, " #other_type " other) -> Tensor(a!)", \
        [](const Node* node) {                                               \
          return [=](Stack& stack) {                                         \
            at::Tensor t;                                                    \
            c_type other;                                                    \
            pop(stack, t, other);                                            \
            std::move(t) = other; /* NOLINT(bugprone-use-after-move) */      \
            push(stack, std::move(t)); /* NOLINT(bugprone-use-after-move) */ \
            return 0;                                                        \
          };                                                                 \
        })

      CREATE_COPY_OP(Tensor, at::Tensor),
      CREATE_COPY_OP(int, int64_t),
      CREATE_COPY_OP(float, double),
  #undef CREATE_COPY_OP

    DEFINE_BINARY_OP(aten::add, a + b),
    DEFINE_BINARY_OP(aten::sub, a - b),
    DEFINE_BINARY_OP(aten::mul, a* b),
    DEFINE_BINARY_OP(aten::pow, static_cast<decltype(a)>(pow(a, b))),
    // min and max are in prim:: because there is a difference between
    // the python builtin 'min' and 'torch.min'
    DEFINE_BINARY_OP(prim::min, a < b ? a : b),
    DEFINE_BINARY_OP(prim::max, a > b ? a : b),
    // Pass in two ops for handling int and float separately as % in C++ only
    // works for int The modulus calculation is different between C++ and Python
    // (on negative), we preserve the python behavior as it's more common and
    // match python syntax, hence the conversion.
    DEFINE_GENERIC_OP(
        aten::remainder,
        (b + (a % b)) % b,
        fmod((b + fmod(a, b)), b),
        int,
        float),
    DEFINE_INT_FLOAT_OP(aten::remainder, fmod((b + fmod(a, b)), b), float),

    DEFINE_GENERIC_OP(
        aten::floordiv,
        floordiv(a, b),
        std::floor(a / b),
        int,
        float),
    DEFINE_INT_FLOAT_OP(aten::floordiv, std::floor(a / b), float),

    // only used in loop unrolling, not exposed to end users
    DEFINE_INT_OP(aten::__round_to_zero_floordiv, a / b),

    DEFINE_INT_OP(aten::__and__, a& b),
    DEFINE_INT_OP(aten::__or__, a | b),
    DEFINE_INT_OP(aten::__xor__, a ^ b),

    // NB: This is the python truediv operation
    Operator(
        "aten::div(int a, int b) -> float",
        [](const Node* node) {
          return [=](Stack& stack) {
            int64_t a, b;
            pop(stack, a, b);
            push(stack, static_cast<double>(a) / static_cast<double>(b));
            return 0;
          };
        }),
    Operator(
        "aten::div(float a, float b) -> float",
        [](const Node* node) {
          return [=](Stack& stack) {
            double a, b;
            pop(stack, a, b);
            push(stack, a / b);
            return 0;
          };
        }),

    Operator(
        "aten::floor(float a) -> int",
        [](const Node* node) {
          return [=](Stack& stack) {
            double a;
            pop(stack, a);
            push(stack, static_cast<int64_t>(std::floor(a)));
            return 0;
          };
        }),

    DEFINE_COMPARISON_OP(aten::ne, a != b),
    DEFINE_COMPARISON_OP(aten::eq, a == b),
    DEFINE_COMPARISON_OP(aten::lt, a < b),
    DEFINE_COMPARISON_OP(aten::gt, a > b),
    DEFINE_COMPARISON_OP(aten::le, a <= b),
    DEFINE_COMPARISON_OP(aten::ge, a >= b),

    DEFINE_BOOL_OP(aten::__and__, a&& b),
    DEFINE_BOOL_OP(aten::__or__, a || b),
    DEFINE_BOOL_OP(aten::__xor__, a != b),

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
            elems.reserve(t.size(0));
            for (int i = 0; i < t.size(0); i++) {
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
            for (size_t i = 0; i < l.size(); i++) {
              t[i] = l[i];
            }
            push(stack, t);
            return 0;
          };
        }),
});

// checking one of size & scale_factor is set
// if scale_factor is a double list check that it's len == dim
// reference: _check_size_scale_factor in torch/nn/functional.py
void _check_size_factor(
    size_t dim,
    const IValue& size,
    const IValue& scale_factor) {
  if (size.isNone() && scale_factor.isNone()) {
    throw std::runtime_error("either size or scale_factor should be defined");
  }
  if (!size.isNone() && !scale_factor.isNone()) {
    throw std::runtime_error(
        "only one of size or scale_factor should be defined");
  }
  if (scale_factor.isDoubleList()) {
    auto scale_len = scale_factor.toDoubleListRef().size();
    if (scale_len != dim) {
      std::stringstream str;
      str << "scale_factor shape must match input shape. Input is " << dim
          << "D, scale_factor size is " << scale_len;
      throw std::runtime_error(
          "only one of size or scale_factor should be defined");
    }
  }
}

// reference: _output_size in torch/nn/functional.py
// size can be none, int or intlist
// scale_factors can be none, float, or floatlist
std::vector<int64_t> _output_size(
    const at::Tensor& input,
    size_t dim,
    const IValue& size,
    const IValue& scale_factors) {
  if (!size.isNone()) {
    if (size.isInt()) {
      std::vector<int64_t> repeated(dim, size.toInt());
      return repeated;
    } else {
      return size.toIntListRef();
    }
  }
  std::vector<double> scale_repeated;
  if (scale_factors.isDouble()) {
    scale_repeated = std::vector<double>(dim, scale_factors.toDouble());
  } else {
    scale_repeated = scale_factors.toDoubleListRef();
  }
  std::vector<int64_t> ret;
  for (size_t i = 0; i < dim; ++i) {
    ret.push_back(std::floor(input.size(i + 2) * scale_repeated[i]));
  }
  return ret;
}

// reference: interpolate in torch/nn/functional.py
// size can be none, int or intlist
// scale_factors can be none, float, or floatlist
at::Tensor interpolate(
    const at::Tensor& input,
    const IValue& size,
    const IValue& scale_factors,
    const std::string& mode,
    c10::optional<bool> align_corners) {
  if ((mode == "nearest" || mode == "area")) {
    if (align_corners != c10::nullopt) {
      throw std::runtime_error(
          "align_corners option can only be set with the "
          "interpolating modes: linear | bilinear | bicubic | trilinear");
    }
  } else {
    if (align_corners == c10::nullopt) {
      AT_WARN(
          "Default upsampling behavior when mode=",
          mode,
          " is changed "
          "to align_corners=False since 0.4.0. Please specify align_corners=True "
          "if the old behavior is desired. See the documentation of nn.Upsample for details");
      align_corners = false;
    }
  }

  auto input_dim = input.dim();
  if (input_dim == 3 && mode == "nearest")
    return at::upsample_nearest1d(
        input, _output_size(input, 1, size, scale_factors));
  if (input_dim == 4 && mode == "nearest")
    return at::upsample_nearest2d(
        input, _output_size(input, 2, size, scale_factors));
  if (input_dim == 5 && mode == "nearest")
    return at::upsample_nearest3d(
        input, _output_size(input, 3, size, scale_factors));
  if (input_dim == 3 && mode == "area")
    return at::adaptive_avg_pool1d(
        input, _output_size(input, 1, size, scale_factors));
  if (input_dim == 4 && mode == "area")
    return at::adaptive_avg_pool2d(
        input, _output_size(input, 2, size, scale_factors));
  if (input_dim == 5 && mode == "area")
    return at::adaptive_avg_pool3d(
        input, _output_size(input, 3, size, scale_factors));
  if (input_dim == 3 && mode == "linear")
    return at::upsample_linear1d(
        input, _output_size(input, 1, size, scale_factors), *align_corners);
  if (input_dim == 3 && mode == "bilinear")
    throw std::runtime_error("Got 3D input, but bilinear mode needs 4D input");
  if (input_dim == 3 && mode == "bicubic")
    throw std::runtime_error("Got 3D input, but bicubic mode needs 4D input");
  if (input_dim == 3 && mode == "trilinear")
    throw std::runtime_error("Got 3D input, but trilinear mode needs 5D input");
  if (input_dim == 4 && mode == "linear")
    throw std::runtime_error("Got 4D input, but linear mode needs 3D input");
  if (input_dim == 4 && mode == "bilinear")
    return at::upsample_bilinear2d(
        input, _output_size(input, 2, size, scale_factors), *align_corners);
  if (input_dim == 4 && mode == "bicubic")
    return at::upsample_bicubic2d(
        input, _output_size(input, 2, size, scale_factors), *align_corners);
  if (input_dim == 4 && mode == "trilinear")
    throw std::runtime_error("Got 4D input, but trilinear mode needs 5D input");
  if (input_dim == 5 && mode == "linear")
    throw std::runtime_error("Got 5D input, but linear mode needs 3D input");
  if (input_dim == 5 && mode == "bilinear")
    throw std::runtime_error("Got 5D input, but bilinear mode needs 4D input");
  if (input_dim == 5 && mode == "bicubic")
    throw std::runtime_error("Got 5D input, but bicubic mode needs 4D input");
  if (input_dim == 5 && mode == "trilinear")
    return at::upsample_trilinear3d(
        input, _output_size(input, 3, size, scale_factors), *align_corners);

  AT_ERROR(
      "Input Error: Only 3D, 4D and 5D input Tensors supported",
      " (got ",
      input_dim,
      "D) for the modes: nearest | linear | bilinear | trilinear",
      " (got ",
      mode,
      ") ");
}

Operation interpolate_op(const Node* n) {
  return [](Stack& stack) {
    at::Tensor input;
    IValue size;
    IValue scale_factors;
    std::string mode;
    IValue align_corners;
    pop(stack, input, size, scale_factors, mode, align_corners);
    at::Tensor res = interpolate(
        input, size, scale_factors, mode, align_corners.toOptional<bool>());
    push(stack, res);
    return 0;
  };
}

// interpolate takes in float & float[] for scale factor
// upsample takes in int & int[], so convert the ints to floats before
// passing on to the interpolate op
IValue convert_scale_factor_to_double(const IValue& int_ivalue) {
  IValue scale_factor_double;
  if (int_ivalue.isInt()) {
    scale_factor_double = static_cast<double>(int_ivalue.toInt());
  } else if (int_ivalue.isIntList()) {
    auto int_list = int_ivalue.toIntListRef();
    std::vector<double> double_vec(int_list.begin(), int_list.end());
    scale_factor_double = double_vec;
  } else if (int_ivalue.isNone()) {
    return IValue();
  } else {
    std::stringstream ss;
    ss << "Expecting optional int or int list arg for scale factor, got"
       << int_ivalue;
    throw std::runtime_error(ss.str());
  }
  return scale_factor_double;
}

Operation upsample_nearest_op(const Node* n) {
  return [](Stack& stack) {
    at::Tensor input;
    IValue size;
    IValue scale_factor_int;
    pop(stack, input, size, scale_factor_int);
    IValue scale_factor_double =
        convert_scale_factor_to_double(scale_factor_int);
    at::Tensor res =
        interpolate(input, size, scale_factor_double, "nearest", c10::nullopt);
    push(stack, res);
    return 0;
  };
}

Operation upsample_op(const Node* n) {
  return [](Stack& stack) {
    at::Tensor input;
    IValue size;
    IValue scale_factor_int;
    std::string mode;
    IValue align_corners;
    pop(stack, input, size, scale_factor_int, mode, align_corners);
    IValue scale_factor_double =
        convert_scale_factor_to_double(scale_factor_int);
    at::Tensor res = interpolate(
        input,
        size,
        scale_factor_double,
        mode,
        align_corners.toOptional<bool>());
    push(stack, res);
    return 0;
  };
}

Operation upsample_bilinear_op(const Node* n) {
  return [](Stack& stack) {
    at::Tensor input;
    IValue size;
    IValue scale_factor_int;
    pop(stack, input, size, scale_factor_int);
    IValue scale_factor_double =
        convert_scale_factor_to_double(scale_factor_int);
    at::Tensor res =
        interpolate(input, size, scale_factor_double, "bilinear", true);
    push(stack, res);
    return 0;
  };
}

RegisterOperators reg3({
    Operator(
        "aten::__interpolate(Tensor input, int? size = None, float[]? scale_factor = None, str mode = 'nearest', bool? align_corners = None) -> Tensor",
        interpolate_op),
    Operator(
        "aten::__interpolate(Tensor input, int[]? size = None, float[]? scale_factor = None, str mode = 'nearest', bool? align_corners = None) -> Tensor",
        interpolate_op),
    Operator(
        "aten::__interpolate(Tensor input, int? size = None, float? scale_factor = None, str mode = 'nearest', bool? align_corners = None) -> Tensor",
        interpolate_op),
    Operator(
        "aten::__interpolate(Tensor input, int[]? size = None, float? scale_factor = None, str mode = 'nearest', bool? align_corners = None) -> Tensor",
        interpolate_op),

    Operator(
        "aten::__upsample_nearest(Tensor input, int? size = None, int? scale_factor = None) -> Tensor",
        upsample_nearest_op),
    Operator(
        "aten::__upsample_nearest(Tensor input, int[]? size = None, int? scale_factor = None) -> Tensor",
        upsample_nearest_op),

    Operator(
        "aten::__upsample(Tensor input, int? size = None, int? scale_factor = None, str mode = 'nearest', bool? align_corners = None) -> Tensor",
        upsample_op),
    Operator(
        "aten::__upsample(Tensor input, int[]? size = None, int? scale_factor = None, str mode = 'nearest', bool? align_corners = None) -> Tensor",
        upsample_op),

    Operator(
        "aten::__upsample_bilinear(Tensor input, int? size = None, int? scale_factor = None) -> Tensor",
        upsample_bilinear_op),
    Operator(
        "aten::__upsample_bilinear(Tensor input, int[]? size = None, int? scale_factor = None) -> Tensor",
        upsample_bilinear_op),
    Operator(
        "aten::__upsample_bilinear(Tensor input, int? size = None, int[]? scale_factor = None) -> Tensor",
        upsample_bilinear_op),
    Operator(
        "aten::__upsample_bilinear(Tensor input, int[]? size = None, int[]? scale_factor = None) -> Tensor",
        upsample_bilinear_op),
});

} // namespace jit
} // namespace torch
