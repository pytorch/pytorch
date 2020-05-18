#pragma once

#include <aten/src/ATen/Context.h>
#include <c10/core/DeviceType.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/csrc/jit/runtime/logging.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/print_handler.h>
#include <torch/csrc/jit/runtime/profiling_record.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>
#include <torch/csrc/jit/serialization/pickle.h>

#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/Dict.h>
#include <ATen/core/ivalue.h>
#include <c10/core/thread_pool.h>
#include <c10/util/SmallVector.h>
#include <c10/util/math_compat.h>
#include <c10/util/string_utils.h>

namespace torch {
namespace jit {
inline c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

inline c10::AliasAnalysisKind aliasAnalysisConservative() {
  return c10::AliasAnalysisKind::CONSERVATIVE;
}

inline c10::AliasAnalysisKind aliasAnalysisSpecialCase() {
  return c10::AliasAnalysisKind::INTERNAL_SPECIAL_CASE;
}

template <class T>
c10::List<T> make_result_list(const TypePtr& elemType) {
  return c10::List<T>();
}

template <>
c10::impl::GenericList make_result_list<IValue>(const TypePtr& elemType);

inline int noop(Stack& n) {
  return 0;
}

// using the rules from python_arg_parser FunctionParameter::check
// tensor cannot have grad set, tensor must be 0 dim,
// and if the dest is an int the source must be integral type
void checkImplicitTensorToNum(at::Tensor t, bool toInt);

// Convert the tensor pointed to by \p data to a nested list. \p dim is the
// number of dimensions in the tensor and \p cur_dim is the dimension being
// processed by the current invocation. \p ty is the expected output IR type of
// the operation. \p is the scalar type of \p data. \p sizes and \p strides are
// the sizes and strides of the tensor operand and \p element_size is the size
// in bytes of one tensor element.
IValue tensorToListRecursive(
    char* data,
    int64_t cur_dim,
    int64_t num_tensor_dims,
    TypePtr ty,
    at::ScalarType scalar_ty,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    size_t element_size);

static int64_t floordiv(int64_t a, int64_t b) {
  if (b == 0) {
    throw std::runtime_error("division by 0");
  }
  if ((a > 0) == (b > 0)) {
    // simple case, both have same sign
    return a / b;
  } else {
    // in python division rounds down, it doesn't not truncate like in c++
    auto r = lldiv(a, b);
    return (r.rem) ? r.quot - 1 : r.quot;
  }
}
TORCH_API void checkDoubleInRange(double a);
static int64_t floor(double a) {
  checkDoubleInRange(a);
  return std::floor(a);
}
static int64_t ceil(double a) {
  checkDoubleInRange(a);
  return std::ceil(a);
}

static int64_t gcd(int64_t a, int64_t b) {
  while (b != 0) {
    int64_t r = a % b;
    a = b;
    b = r;
  }
  // in python gcd returns non-negative values
  return std::abs(a);
}

int64_t partProduct(int n, int m);

void loop(int n, int64_t& p, int64_t& r);

int nminussumofbits(int v);

int64_t factorial(int n);
static const double degToRad = std::acos(-1.0) / 180.0;
static const double radToDeg = 180.0 / std::acos(-1.0);
double degrees(double x);
double radians(double x);

// reference function THPVariable_to in python_variable_methods.cpp
static at::Tensor to_dispatch(
    at::Tensor self,
    c10::optional<at::Device> device,
    c10::optional<at::ScalarType> scalarType,
    bool non_blocking,
    bool copy) {
  if (device && device->is_cuda()) {
    at::globalContext().lazyInitCUDA();
  }
  if (!device && !scalarType && !copy) {
    return self;
  } else if (!device) {
    return self.to(*scalarType, non_blocking, copy);
  } else if (!scalarType) {
    return self.to(*device, non_blocking, copy);
  } else {
    return self.to(*device, *scalarType, non_blocking, copy);
  }
}

// Convert an python index (which may be negative) into an index usable for a
// C++ container
int64_t normalizeIndex(int64_t idx, int64_t list_size);

// Equivalent to list.at(idx)
template <typename T>
T getItem(const c10::List<T>& list, int64_t idx) {
  const int64_t list_size = list.size();
  const int64_t normalized_idx = normalizeIndex(idx, list_size);
  if (normalized_idx < 0 || normalized_idx >= list_size) {
    throw std::out_of_range("list index out of range");
  }
  return list.get(normalized_idx);
}

template <typename T>
void setItem(const c10::List<T>& list, int64_t idx, T&& value) {
  const int64_t list_size = list.size();
  const int64_t normalized_idx = normalizeIndex(idx, list_size);
  if (normalized_idx < 0 || normalized_idx >= list_size) {
    throw std::out_of_range("list index out of range");
  }
  list.set(normalized_idx, std::move(value));
}

int listAppend(Stack& stack);

int listReverse(Stack& stack);

template <typename T>
int minList(Stack& stack) {
  c10::List<T> a = pop(stack).to<c10::List<T>>();
  c10::List<T> b = pop(stack).to<c10::List<T>>();

  size_t min_size = std::min(a.size(), b.size());
  for (size_t i = 0; i < min_size; i++) {
    if (a[i] == b[i]) {
      continue;
    }

    push(stack, a[i] < b[i] ? a : b);
    return 0;
  }

  push(stack, b.size() < a.size() ? b : a);
  return 0;
}

template <typename T>
int maxList(Stack& stack) {
  c10::List<T> a = pop(stack).to<c10::List<T>>();
  c10::List<T> b = pop(stack).to<c10::List<T>>();

  size_t min_size = std::min(a.size(), b.size());
  for (size_t i = 0; i < min_size; i++) {
    if (a[i] == b[i]) {
      continue;
    }

    push(stack, a[i] > b[i] ? a : b);
    return 0;
  }

  push(stack, b.size() > a.size() ? b : a);
  return 0;
}

int listPopImpl(Stack& stack, const char* empty_message);

int listPop(Stack& stack);

int listClear(Stack& stack);

int listDelete(Stack& stack);

int listInsert(Stack& stack);

template <typename T>
int listRemove(Stack& stack) {
  T elem = pop(stack).to<T>();
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  auto pos = std::find(list.begin(), list.end(), elem);

  if (pos != list.end()) {
    list.erase(pos);
  } else {
    AT_ERROR("list.remove(x): x not in list");
  }

  return 0;
}

template <typename T>
int listMin(Stack& stack) {
  c10::List<T> list = pop(stack).to<c10::List<T>>();
  size_t list_size = list.size();
  if (list_size == 0) {
    throw std::runtime_error("min() arg is an empty sequence");
  }

  T min_elem = list[0];
  for (size_t i = 1; i < list_size; ++i) {
    T elem = list[i];
    min_elem = elem < min_elem ? elem : min_elem;
  }

  stack.push_back(min_elem);
  return 0;
}

template <typename T>
int listMax(Stack& stack) {
  c10::List<T> list = pop(stack).to<c10::List<T>>();
  size_t list_size = list.size();
  if (list_size == 0) {
    throw std::runtime_error("max() arg is an empty sequence");
  }

  T max_elem = list[0];
  for (size_t i = 1; i < list_size; ++i) {
    T elem = list[i];
    max_elem = elem > max_elem ? elem : max_elem;
  }

  stack.push_back(max_elem);
  return 0;
}

template <>
int listRemove<at::Tensor>(Stack& stack);

template <typename T>
int listIndex(Stack& stack) {
  T elem = pop(stack).to<T>();
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  auto pos = std::find(list.begin(), list.end(), elem);

  if (pos != list.end()) {
    push(stack, static_cast<int64_t>(std::distance(list.begin(), pos)));
  } else {
    AT_ERROR("'", elem, "' is not in list");
  }

  return 0;
}

template <>
int listIndex<at::Tensor>(Stack& stack);

template <typename T>
int listCount(Stack& stack) {
  T elem = pop(stack).to<T>();
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  const int64_t count = std::count(list.begin(), list.end(), elem);
  push(stack, count);

  return 0;
}

template <>
int listCount<at::Tensor>(Stack& stack);

int listExtend(Stack& stack);

int listCopy(Stack& stack);

int listSelect(Stack& stack);

int listLen(Stack& stack);

template <typename T>
int listEq(Stack& stack) {
  c10::List<T> b = pop(stack).to<c10::List<T>>();
  c10::List<T> a = pop(stack).to<c10::List<T>>();
  push(stack, a == b);
  return 0;
}

template <typename T>
int listNe(Stack& stack) {
  c10::List<T> b = pop(stack).to<c10::List<T>>();
  c10::List<T> a = pop(stack).to<c10::List<T>>();
  push(stack, a != b);
  return 0;
}

inline bool tensor_list_equal(
    const c10::List<at::Tensor>& a,
    const c10::List<at::Tensor>& b) {
  if (a.size() != b.size()) {
    return false;
  }

  for (size_t i = 0; i < a.size(); ++i) {
    at::Tensor a_element = a[i];
    at::Tensor b_element = b[i];
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
int listEq<at::Tensor>(Stack& stack);

// Specialization for at::Tensor, since it doesn't define operator==
template <>
int listNe<at::Tensor>(Stack& stack);

int listList(Stack& stack);

template <typename T>
int listContains(Stack& stack) {
  auto key = pop(stack).to<T>();
  auto list = pop(stack).to<c10::List<T>>();
  for (const T& item : list) {
    if (item == key) {
      push(stack, true);
      return 0;
    }
  }
  push(stack, false);
  return 0;
}

int listAdd(Stack& stack);

int listInplaceAdd(Stack& stack);

int listMulIntLeftInPlace(Stack& stack);

int listMulIntLeft(Stack& stack);

int listMulIntRight(Stack& stack);

int listSlice(Stack& stack);

template <typename T>
int listSort(Stack& stack) {
  bool reverse = pop(stack).toBool();
  c10::List<T> list = pop(stack).to<c10::List<T>>();
  std::sort(list.begin(), list.end(), [reverse](const T& a, const T& b) {
    // FBCode errors without this check - "strict weak ordering"
    // TODO: remove when possible, since it just slows down
    // sorting and doesn't do anything useful
    if (a == b) {
      return false;
    }
    return (a < b) != reverse;
  });
  return 0;
}

// Specialization for at::Tensor
template <>
int listSort<at::Tensor>(Stack& stack);

template <typename T>
int listCopyAndSort(Stack& stack) {
  c10::List<T> list = pop(stack).to<c10::List<T>>();
  auto list_copied = list.copy();
  std::sort(list_copied.begin(), list_copied.end(), [](const T& a, const T& b) {
    // "strict weak ordering" issue - see other sort
    if (a == b) {
      return false;
    }
    return a < b;
  });
  push(stack, list_copied);
  return 0;
}

// Specialization for at::Tensor
template <>
int listCopyAndSort<at::Tensor>(Stack& stack);

int listSetItem(Stack& stack);

// define implementations for primitive number ops
#define DEFINE_GENERIC_OP(aten_op, int_op, float_op, int_result, float_result) \
  Operator(                                                                    \
      #aten_op ".int(int a, int b) -> " #int_result,                           \
      [](Stack& stack) {                                                       \
        int64_t a, b;                                                          \
        pop(stack, a, b);                                                      \
        push(stack, int_op);                                                   \
        return 0;                                                              \
      },                                                                       \
      aliasAnalysisFromSchema()),                                              \
      Operator(                                                                \
          #aten_op ".float(float a, float b) -> " #float_result,               \
          [](Stack& stack) {                                                   \
            double a, b;                                                       \
            pop(stack, a, b);                                                  \
            push(stack, float_op);                                             \
            return 0;                                                          \
          },                                                                   \
          aliasAnalysisFromSchema())

#define DEFINE_INT_FLOAT_OP(aten_op, op, result)             \
  Operator(                                                  \
      #aten_op ".int_float(int a, float b) -> " #result,     \
      [](Stack& stack) {                                     \
        int64_t a;                                           \
        double b;                                            \
        pop(stack, a, b);                                    \
        push(stack, op);                                     \
        return 0;                                            \
      },                                                     \
      aliasAnalysisFromSchema()),                            \
      Operator(                                              \
          #aten_op ".float_int(float a, int b) -> " #result, \
          [](Stack& stack) {                                 \
            double a;                                        \
            int64_t b;                                       \
            pop(stack, a, b);                                \
            push(stack, op);                                 \
            return 0;                                        \
          },                                                 \
          aliasAnalysisFromSchema())

#define DEFINE_INT_OP(aten_op, op)                          \
  Operator(                                                 \
      #aten_op "(int a, int b) -> int",                     \
      [](Stack& stack) {                                    \
        int64_t a, b;                                       \
        pop(stack, a, b);                                   \
        push(stack, op); /* NOLINT(hicpp-signed-bitwise) */ \
        return 0;                                           \
      },                                                    \
      aliasAnalysisFromSchema())

#define DEFINE_STR_CMP_OP(aten_op, op)     \
  Operator(                                \
      #aten_op "(str a, str b) -> bool",   \
      [](Stack& stack) {                   \
        auto b = pop(stack).toStringRef(); \
        auto a = pop(stack).toStringRef(); \
        push(stack, op);                   \
        return 0;                          \
      },                                   \
      aliasAnalysisFromSchema())

// define a primitive op over Scalar operands.
// it's necessary to register this overload following
// int/float variations to avoid trapping Scalar args
// in unintended implicit conversions
#define DEFINE_SCALAR_BINARY_OP(aten_op, int_op, float_op, result) \
  Operator(                                                        \
      #aten_op "(Scalar a, Scalar b) -> " #result,                 \
      [](Stack& stack) {                                           \
        IValue x, y;                                               \
        pop(stack, x, y);                                          \
        if (x.isDouble()) {                                        \
          if (y.isDouble()) {                                      \
            double a = x.toDouble();                               \
            double b = y.toDouble();                               \
            push(stack, float_op);                                 \
          } else {                                                 \
            double a = x.toDouble();                               \
            int64_t b = y.toInt();                                 \
            push(stack, float_op);                                 \
          }                                                        \
        } else {                                                   \
          if (y.isDouble()) {                                      \
            int64_t a = x.toInt();                                 \
            double b = y.toDouble();                               \
            push(stack, float_op);                                 \
          } else {                                                 \
            int64_t a = x.toInt();                                 \
            int64_t b = y.toInt();                                 \
            push(stack, int_op);                                   \
          }                                                        \
        }                                                          \
        return 0;                                                  \
      },                                                           \
      aliasAnalysisFromSchema())

#define DEFINE_BINARY_OP(aten_op, op)             \
  DEFINE_GENERIC_OP(aten_op, op, op, int, float), \
      DEFINE_INT_FLOAT_OP(aten_op, op, float),    \
      DEFINE_SCALAR_BINARY_OP(aten_op, op, op, Scalar)

#define DEFINE_BINARY_FLOAT_OP(aten_op, op)         \
  DEFINE_GENERIC_OP(aten_op, op, op, float, float), \
      DEFINE_INT_FLOAT_OP(aten_op, op, float),      \
      DEFINE_SCALAR_BINARY_OP(aten_op, op, op, float)

#define DEFINE_COMPARISON_OP(aten_op, op)             \
  DEFINE_GENERIC_OP(aten_op, op, op, bool, bool),     \
      DEFINE_INT_FLOAT_OP(aten_op, op, bool),         \
      DEFINE_SCALAR_BINARY_OP(aten_op, op, op, bool), \
      DEFINE_STR_CMP_OP(aten_op, op)

#define DEFINE_UNARY_INT_OP(aten_op, op, result) \
  Operator(                                      \
      #aten_op ".int(int a) -> " #result,        \
      [](Stack& stack) {                         \
        int64_t a;                               \
        pop(stack, a);                           \
        push(stack, op);                         \
        return 0;                                \
      },                                         \
      aliasAnalysisFromSchema())

#define DEFINE_UNARY_FLOAT_OP(aten_op, op, result) \
  Operator(                                        \
      #aten_op ".float(float a) -> " #result,      \
      [](Stack& stack) {                           \
        double a;                                  \
        pop(stack, a);                             \
        push(stack, op);                           \
        return 0;                                  \
      },                                           \
      aliasAnalysisFromSchema())

#define DEFINE_UNARY_OP(aten_op, op, int_result, float_result) \
  DEFINE_UNARY_INT_OP(aten_op, op, int_result),                \
      DEFINE_UNARY_FLOAT_OP(aten_op, op, float_result),        \
      Operator(                                                \
          #aten_op ".Scalar(Scalar a) -> Scalar",              \
          [](Stack& stack) {                                   \
            IValue x;                                          \
            pop(stack, x);                                     \
            if (x.isDouble()) {                                \
              double a = x.toDouble();                         \
              push(stack, static_cast<float_result>(op));      \
            } else {                                           \
              int64_t a = x.toInt();                           \
              push(stack, static_cast<int_result>(op));        \
            }                                                  \
            return 0;                                          \
          },                                                   \
          aliasAnalysisFromSchema())
#define DEFINE_BOOL_OP(aten_op, op)        \
  Operator(                                \
      #aten_op "(bool a, bool b) -> bool", \
      [](Stack& stack) {                   \
        bool a, b;                         \
        pop(stack, a, b);                  \
        push(stack, op);                   \
        return 0;                          \
      },                                   \
      aliasAnalysisFromSchema())

} // namespace jit
} // namespace torch
