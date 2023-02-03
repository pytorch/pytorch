#pragma once

#include <ATen/Context.h>
#include <c10/core/DeviceType.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/mobile/register_ops_common_utils.h>
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
#include <c10/util/irange.h>
#include <c10/util/math_compat.h>
#include <c10/util/string_utils.h>

namespace torch {
namespace jit {
constexpr inline c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

constexpr inline c10::AliasAnalysisKind aliasAnalysisConservative() {
  return c10::AliasAnalysisKind::CONSERVATIVE;
}

constexpr inline c10::AliasAnalysisKind aliasAnalysisSpecialCase() {
  return c10::AliasAnalysisKind::INTERNAL_SPECIAL_CASE;
}

template <class T>
c10::List<T> make_result_list(const TypePtr& elemType) {
  return c10::List<T>();
}

template <>
c10::impl::GenericList make_result_list<IValue>(const TypePtr& elemType);

// As described in https://docs.python.org/3/library/functions.html#round
// When a number is exactly halfway between two integers, python builtin round
// function will round to even number. We use round(x/2)*2 to handle the
// special halfway case. For positive 'x', round(x/2)*2 =
// round((x_e + x_r)/2)*2 = x_e + round(x_r/2)*2, where x_e is an even integer,
// x_r is either 0.5 of 1.5, round(x_r/2)*2 results a 0 or 2, so the final
// result will always be a even number. Due to symmetricity, it also applies to
// negative cases.
inline double round_to_even(double a) {
  return a - std::floor(a) == 0.5 ? (std::round(a * 0.5) * 2.0) : std::round(a);
}

// using the rules from python_arg_parser FunctionParameter::check
// tensor cannot have grad set, tensor must be 0 dim,
// and if the dest is an int the source must be integral type
void checkImplicitTensorToNum(const at::Tensor& t, bool toInt);

static C10_UNUSED int64_t floordiv(int64_t a, int64_t b) {
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
static C10_UNUSED int64_t floor(double a) {
  checkDoubleInRange(a);
  return std::floor(a);
}
static C10_UNUSED int64_t ceil(double a) {
  checkDoubleInRange(a);
  return std::ceil(a);
}

static C10_UNUSED int64_t gcd(int64_t a, int64_t b) {
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

// Convert an python index (which may be negative) into an index usable for a
// C++ container

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
  list.set(normalized_idx, std::forward<T>(value));
}

void listAppend(Stack& stack);

void listReverse(Stack& stack);

template <typename T>
void minList(Stack& stack) {
  c10::List<T> a = pop(stack).to<c10::List<T>>();
  c10::List<T> b = pop(stack).to<c10::List<T>>();

  size_t min_size = std::min(a.size(), b.size());
  for (const auto i : c10::irange(min_size)) {
    if (a[i] == b[i]) {
      continue;
    }

    push(stack, a[i] < b[i] ? a : b);
    return;
  }

  push(stack, b.size() < a.size() ? b : a);
}

template <typename T>
void maxList(Stack& stack) {
  c10::List<T> a = pop(stack).to<c10::List<T>>();
  c10::List<T> b = pop(stack).to<c10::List<T>>();

  size_t min_size = std::min(a.size(), b.size());
  for (const auto i : c10::irange(min_size)) {
    if (a[i] == b[i]) {
      continue;
    }

    push(stack, a[i] > b[i] ? a : b);
    return;
  }

  push(stack, b.size() > a.size() ? b : a);
}

void listPopImpl(Stack& stack, const char* empty_message);

void listPop(Stack& stack);

void listClear(Stack& stack);

void listDelete(Stack& stack);

void listInsert(Stack& stack);

template <typename T>
void listRemove(Stack& stack) {
  T elem = pop(stack).to<T>();
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  auto pos = std::find(list.begin(), list.end(), elem);

  if (pos != list.end()) {
    list.erase(pos);
  } else {
    AT_ERROR("list.remove(x): x not in list");
  }
}

template <typename T>
void listMin(Stack& stack) {
  c10::List<T> list = pop(stack).to<c10::List<T>>();
  size_t list_size = list.size();
  if (list_size == 0) {
    throw std::runtime_error("min() arg is an empty sequence");
  }

  T min_elem = list[0];
  for (const auto i : c10::irange(1, list_size)) {
    T elem = list[i];
    min_elem = elem < min_elem ? elem : min_elem;
  }

  stack.push_back(min_elem);
}

template <typename T>
void listMax(Stack& stack) {
  c10::List<T> list = pop(stack).to<c10::List<T>>();
  size_t list_size = list.size();
  if (list_size == 0) {
    throw std::runtime_error("max() arg is an empty sequence");
  }

  T max_elem = list[0];
  for (const auto i : c10::irange(1, list_size)) {
    T elem = list[i];
    max_elem = elem > max_elem ? elem : max_elem;
  }

  stack.push_back(max_elem);
}

template <>
void listRemove<at::Tensor>(Stack& stack);

template <typename T>
void listIndex(Stack& stack) {
  T elem = pop(stack).to<T>();
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  auto pos = std::find(list.begin(), list.end(), elem);

  if (pos != list.end()) {
    push(stack, static_cast<int64_t>(std::distance(list.begin(), pos)));
  } else {
    AT_ERROR("'", elem, "' is not in list");
  }
}

template <>
void listIndex<at::Tensor>(Stack& stack);

template <typename T>
void listCount(Stack& stack) {
  T elem = pop(stack).to<T>();
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  const int64_t count = std::count(list.begin(), list.end(), elem);
  push(stack, count);
}

template <>
void listCount<at::Tensor>(Stack& stack);

void listExtend(Stack& stack);

void listCopy(Stack& stack);

void listSelect(Stack& stack);

void listLen(Stack& stack);

template <typename T>
void listEq(Stack& stack) {
  c10::List<T> b = pop(stack).to<c10::List<T>>();
  c10::List<T> a = pop(stack).to<c10::List<T>>();
  push(stack, a == b);
}

template <typename T>
void listNe(Stack& stack) {
  c10::List<T> b = pop(stack).to<c10::List<T>>();
  c10::List<T> a = pop(stack).to<c10::List<T>>();
  push(stack, a != b);
}

inline bool tensor_list_equal(
    const c10::List<at::Tensor>& a,
    const c10::List<at::Tensor>& b) {
  if (a.size() != b.size()) {
    return false;
  }

  for (const auto i : c10::irange(a.size())) {
    const at::Tensor& a_element = a[i];
    const at::Tensor& b_element = b[i];
    // This preserves Python's semantics, which uses eq() to compare two
    // elements, then passes the result to bool().
    // see: https://docs.python.org/3.4/reference/datamodel.html#object.__ge__
    const auto cmp_result = a_element.eq(b_element);
    if (!at::native::is_nonzero(cmp_result)) {
      return false;
    }
  }

  return true;
}

// Specialization for at::Tensor, since it doesn't define operator==
template <>
void listEq<at::Tensor>(Stack& stack);

// Specialization for at::Tensor, since it doesn't define operator==
template <>
void listNe<at::Tensor>(Stack& stack);

void listList(Stack& stack);

template <typename T>
void listContains(Stack& stack) {
  auto key = pop(stack).to<T>();
  auto list = pop(stack).to<c10::List<T>>();
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const T& item : list) {
    if (item == key) {
      push(stack, true);
      return;
    }
  }
  push(stack, false);
}

void listAdd(Stack& stack);

void listInplaceAdd(Stack& stack);

void listMulIntLeftInPlace(Stack& stack);

void listMulIntLeft(Stack& stack);

void listMulIntRight(Stack& stack);

void listSlice(Stack& stack);

template <typename T>
void listSort(Stack& stack) {
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
}

// Specialization for at::Tensor
template <>
void listSort<at::Tensor>(Stack& stack);

template <typename T>
void listCopyAndSort(Stack& stack) {
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
}

// Specialization for at::Tensor
template <>
void listCopyAndSort<at::Tensor>(Stack& stack);

void listSetItem(Stack& stack);

struct OperatorGeneratorArgs {
  const char* schema_str;
  bool isOperationCreator;
  union {
    void (*operation)(Stack&);
    OperationCreator operationCreator;
  };
  AliasAnalysisKind aliasAnalysis;

  explicit constexpr OperatorGeneratorArgs(
      torch::detail::SelectiveStr<true> schema_str,
      void (*op)(Stack&),
      AliasAnalysisKind aa)
      : schema_str(schema_str),
        isOperationCreator(false),
        operation(op),
        aliasAnalysis(aa) {}

  explicit constexpr OperatorGeneratorArgs(
      torch::detail::SelectiveStr<true> schema_str,
      OperationCreator opCreator,
      AliasAnalysisKind aa)
      : schema_str(schema_str),
        isOperationCreator(true),
        operationCreator(opCreator),
        aliasAnalysis(aa) {}

  template <typename... Args>
  explicit constexpr OperatorGeneratorArgs(
      torch::detail::SelectiveStr<false>,
      Args...)
      : schema_str(nullptr),
        isOperationCreator(false),
        operation(nullptr),
        aliasAnalysis(AliasAnalysisKind::INTERNAL_SPECIAL_CASE) {}
};

#define DEFINE_GENERIC_BINARY_OP(                                             \
    aten_op, op, int_float_result, complex_result)                            \
  OperatorGeneratorArgs(                                                      \
      TORCH_SELECTIVE_SCHEMA(#aten_op                                         \
                             ".int_int(int a, int b) -> " #int_float_result), \
      [](Stack& stack) {                                                      \
        int64_t a, b;                                                         \
        pop(stack, a, b);                                                     \
        push(stack, op);                                                      \
      },                                                                      \
      aliasAnalysisFromSchema()),                                             \
      OperatorGeneratorArgs(                                                  \
          TORCH_SELECTIVE_SCHEMA(                                             \
              #aten_op                                                        \
              ".float_float(float a, float b) -> " #int_float_result),        \
          [](Stack& stack) {                                                  \
            double a, b;                                                      \
            pop(stack, a, b);                                                 \
            push(stack, op);                                                  \
          },                                                                  \
          aliasAnalysisFromSchema()),                                         \
      OperatorGeneratorArgs(                                                  \
          TORCH_SELECTIVE_SCHEMA(                                             \
              #aten_op                                                        \
              ".complex_complex(complex a, complex b) -> " #complex_result),  \
          [](Stack& stack) {                                                  \
            c10::complex<double> a, b;                                        \
            pop(stack, a, b);                                                 \
            push(stack, op);                                                  \
          },                                                                  \
          aliasAnalysisFromSchema())

// define implementations for primitive number ops
#define DEFINE_GENERIC_OP(aten_op, int_op, float_op, int_result, float_result) \
  OperatorGeneratorArgs(                                                       \
      TORCH_SELECTIVE_SCHEMA(#aten_op ".int(int a, int b) -> " #int_result),   \
      [](Stack& stack) {                                                       \
        int64_t a, b;                                                          \
        pop(stack, a, b);                                                      \
        push(stack, int_op);                                                   \
      },                                                                       \
      aliasAnalysisFromSchema()),                                              \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA(                                              \
              #aten_op ".float(float a, float b) -> " #float_result),          \
          [](Stack& stack) {                                                   \
            double a, b;                                                       \
            pop(stack, a, b);                                                  \
            push(stack, float_op);                                             \
          },                                                                   \
          aliasAnalysisFromSchema())

#define DEFINE_INT_FLOAT_OP(aten_op, op, result)                            \
  OperatorGeneratorArgs(                                                    \
      TORCH_SELECTIVE_SCHEMA(#aten_op                                       \
                             ".int_float(int a, float b) -> " #result),     \
      [](Stack& stack) {                                                    \
        int64_t a;                                                          \
        double b;                                                           \
        pop(stack, a, b);                                                   \
        push(stack, op);                                                    \
      },                                                                    \
      aliasAnalysisFromSchema()),                                           \
      OperatorGeneratorArgs(                                                \
          TORCH_SELECTIVE_SCHEMA(#aten_op                                   \
                                 ".float_int(float a, int b) -> " #result), \
          [](Stack& stack) {                                                \
            double a;                                                       \
            int64_t b;                                                      \
            pop(stack, a, b);                                               \
            push(stack, op);                                                \
          },                                                                \
          aliasAnalysisFromSchema())

#define DEFINE_INT_OP(aten_op, op)                                  \
  OperatorGeneratorArgs(                                            \
      TORCH_SELECTIVE_SCHEMA(#aten_op ".int(int a, int b) -> int"), \
      [](Stack& stack) {                                            \
        int64_t a, b;                                               \
        pop(stack, a, b);                                           \
        push(stack, op); /* NOLINT(hicpp-signed-bitwise) */         \
      },                                                            \
      aliasAnalysisFromSchema())

#define DEFINE_STR_CMP_OP(aten_op, op)                               \
  OperatorGeneratorArgs(                                             \
      TORCH_SELECTIVE_SCHEMA(#aten_op ".str(str a, str b) -> bool"), \
      [](Stack& stack) {                                             \
        auto b = pop(stack).toStringRef();                           \
        auto a = pop(stack).toStringRef();                           \
        push(stack, op);                                             \
      },                                                             \
      aliasAnalysisFromSchema())

// define a primitive op over Scalar operands.
// it's necessary to register this overload following
// int/float variations to avoid trapping Scalar args
// in unintended implicit conversions
#define DEFINE_SCALAR_BINARY_OP_AVOID_COLLISION_GENERIC(          \
    aten_op, int_op, float_op, result, string_val)                \
  OperatorGeneratorArgs(                                          \
      TORCH_SELECTIVE_SCHEMA(#aten_op string_val                  \
                             "(Scalar a, Scalar b) -> " #result), \
      [](Stack& stack) {                                          \
        IValue x, y;                                              \
        pop(stack, x, y);                                         \
        if (x.isDouble()) {                                       \
          if (y.isDouble()) {                                     \
            double a = x.toDouble();                              \
            double b = y.toDouble();                              \
            push(stack, float_op);                                \
          } else {                                                \
            double a = x.toDouble();                              \
            int64_t b = y.toInt();                                \
            push(stack, float_op);                                \
          }                                                       \
        } else {                                                  \
          if (y.isDouble()) {                                     \
            int64_t a = x.toInt();                                \
            double b = y.toDouble();                              \
            push(stack, float_op);                                \
          } else {                                                \
            int64_t a = x.toInt();                                \
            int64_t b = y.toInt();                                \
            push(stack, int_op);                                  \
          }                                                       \
        }                                                         \
      },                                                          \
      aliasAnalysisFromSchema())

#define DEFINE_SCALAR_BINARY_OP(aten_op, int_op, float_op, result) \
  DEFINE_SCALAR_BINARY_OP_AVOID_COLLISION_GENERIC(                 \
      aten_op, int_op, float_op, result, "")

#define DEFINE_SCALAR_BINARY_OP_AVOID_COLLISION(   \
    aten_op, int_op, float_op, result)             \
  DEFINE_SCALAR_BINARY_OP_AVOID_COLLISION_GENERIC( \
      aten_op, int_op, float_op, result, ".Scalar_Scalar")

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

#define DEFINE_UNARY_INT_OP(aten_op, op, result)                  \
  OperatorGeneratorArgs(                                          \
      TORCH_SELECTIVE_SCHEMA(#aten_op ".int(int a) -> " #result), \
      [](Stack& stack) {                                          \
        int64_t a;                                                \
        pop(stack, a);                                            \
        push(stack, op);                                          \
      },                                                          \
      aliasAnalysisFromSchema())

#define DEFINE_UNARY_FLOAT_OP(aten_op, op, result)                    \
  OperatorGeneratorArgs(                                              \
      TORCH_SELECTIVE_SCHEMA(#aten_op ".float(float a) -> " #result), \
      [](Stack& stack) {                                              \
        double a;                                                     \
        pop(stack, a);                                                \
        push(stack, op);                                              \
      },                                                              \
      aliasAnalysisFromSchema())

#define DEFINE_UNARY_OP(aten_op, op, int_result, float_result)            \
  DEFINE_UNARY_INT_OP(aten_op, op, int_result),                           \
      DEFINE_UNARY_FLOAT_OP(aten_op, op, float_result),                   \
      OperatorGeneratorArgs(                                              \
          TORCH_SELECTIVE_SCHEMA(#aten_op ".Scalar(Scalar a) -> Scalar"), \
          [](Stack& stack) {                                              \
            IValue x;                                                     \
            pop(stack, x);                                                \
            if (x.isDouble()) {                                           \
              double a = x.toDouble();                                    \
              push(stack, static_cast<float_result>(op));                 \
            } else {                                                      \
              int64_t a = x.toInt();                                      \
              push(stack, static_cast<int_result>(op));                   \
            }                                                             \
          },                                                              \
          aliasAnalysisFromSchema())
#define DEFINE_BOOL_OP(aten_op, op)                                     \
  OperatorGeneratorArgs(                                                \
      TORCH_SELECTIVE_SCHEMA(#aten_op ".bool(bool a, bool b) -> bool"), \
      [](Stack& stack) {                                                \
        bool a, b;                                                      \
        pop(stack, a, b);                                               \
        push(stack, op);                                                \
      },                                                                \
      aliasAnalysisFromSchema())
#define DEFINE_STRING_OP(op_name, string_op, result)                    \
  OperatorGeneratorArgs(                                                \
      TORCH_SELECTIVE_SCHEMA(#op_name ".str(str a, str b) ->" #result), \
      [](Stack& stack) {                                                \
        auto b = pop(stack).toStringRef();                              \
        auto a = pop(stack).toStringRef();                              \
        push(stack, string_op);                                         \
      },                                                                \
      aliasAnalysisFromSchema())

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
#define DEFINE_UNARY_COMPLEX_OP(aten_op, op, result)                      \
  OperatorGeneratorArgs(                                                  \
      TORCH_SELECTIVE_SCHEMA(#aten_op ".complex(complex a) -> " #result), \
      [](Stack& stack) {                                                  \
        c10::complex<double> a;                                           \
        pop(stack, a);                                                    \
        push(stack, op);                                                  \
      },                                                                  \
      aliasAnalysisFromSchema())

// Some complex unary ops (like abs, angle) return real valued output, but most
// other unary ops return complex valued output. So, this macro is used in the
// former case where we can explicitly pass complex_result_cast argument, which
// is set to c10::complex<float> in the macro `DEFINE_UNARY_OP_WITH_COMPLEX`
// defined below.
#define DEFINE_UNARY_OP_WITH_COMPLEX_CAST(                                \
    aten_op,                                                              \
    op,                                                                   \
    int_result,                                                           \
    float_result,                                                         \
    complex_result,                                                       \
    complex_result_cast)                                                  \
  DEFINE_UNARY_INT_OP(aten_op, op, int_result),                           \
      DEFINE_UNARY_FLOAT_OP(aten_op, op, float_result),                   \
      DEFINE_UNARY_COMPLEX_OP(aten_op, op, complex_result),               \
      OperatorGeneratorArgs(                                              \
          TORCH_SELECTIVE_SCHEMA(#aten_op ".Scalar(Scalar a) -> Scalar"), \
          [](Stack& stack) {                                              \
            IValue x;                                                     \
            pop(stack, x);                                                \
            if (x.isDouble()) {                                           \
              double a = x.toDouble();                                    \
              push(stack, static_cast<float_result>(op));                 \
            } else if (x.isComplexDouble()) {                             \
              c10::complex<double> a = x.toComplexDouble();               \
              push(stack, static_cast<complex_result_cast>(op));          \
            } else {                                                      \
              int64_t a = x.toInt();                                      \
              push(stack, static_cast<int_result>(op));                   \
            }                                                             \
          },                                                              \
          aliasAnalysisFromSchema())

#define DEFINE_UNARY_OP_WITH_COMPLEX(aten_op, op, int_result, float_result) \
  DEFINE_UNARY_OP_WITH_COMPLEX_CAST(                                        \
      aten_op, op, int_result, float_result, complex, c10::complex<double>)

#define DEFINE_GENERIC_OP_WITH_COMPLEX(                                       \
    aten_op,                                                                  \
    int_op,                                                                   \
    float_op,                                                                 \
    complex_op,                                                               \
    int_result,                                                               \
    float_result,                                                             \
    complex_result)                                                           \
  OperatorGeneratorArgs(                                                      \
      TORCH_SELECTIVE_SCHEMA(#aten_op ".int(int a, int b) -> " #int_result),  \
      [](Stack& stack) {                                                      \
        int64_t a, b;                                                         \
        pop(stack, a, b);                                                     \
        push(stack, int_op);                                                  \
      },                                                                      \
      aliasAnalysisFromSchema()),                                             \
      OperatorGeneratorArgs(                                                  \
          TORCH_SELECTIVE_SCHEMA(                                             \
              #aten_op ".complex(complex a, complex b) -> " #complex_result), \
          [](Stack& stack) {                                                  \
            c10::complex<double> a, b;                                        \
            pop(stack, a, b);                                                 \
            push(stack, complex_op);                                          \
          },                                                                  \
          aliasAnalysisFromSchema()),                                         \
      OperatorGeneratorArgs(                                                  \
          TORCH_SELECTIVE_SCHEMA(                                             \
              #aten_op ".float(float a, float b) -> " #float_result),         \
          [](Stack& stack) {                                                  \
            double a, b;                                                      \
            pop(stack, a, b);                                                 \
            push(stack, float_op);                                            \
          },                                                                  \
          aliasAnalysisFromSchema())

#define DEFINE_INT_COMPLEX_OP(aten_op, op, result)                          \
  OperatorGeneratorArgs(                                                    \
      TORCH_SELECTIVE_SCHEMA(#aten_op                                       \
                             ".int_complex(int a, complex b) -> " #result), \
      [](Stack& stack) {                                                    \
        int64_t a;                                                          \
        c10::complex<double> b;                                             \
        pop(stack, a, b);                                                   \
        push(stack, op);                                                    \
      },                                                                    \
      aliasAnalysisFromSchema()),                                           \
      OperatorGeneratorArgs(                                                \
          TORCH_SELECTIVE_SCHEMA(                                           \
              #aten_op ".complex_int(complex a, int b) -> " #result),       \
          [](Stack& stack) {                                                \
            c10::complex<double> a;                                         \
            int64_t b;                                                      \
            pop(stack, a, b);                                               \
            push(stack, op);                                                \
          },                                                                \
          aliasAnalysisFromSchema())

#define DEFINE_FLOAT_COMPLEX_OP(aten_op, op, result)                      \
  OperatorGeneratorArgs(                                                  \
      TORCH_SELECTIVE_SCHEMA(                                             \
          #aten_op ".float_complex(float a, complex b) -> " #result),     \
      [](Stack& stack) {                                                  \
        double a;                                                         \
        c10::complex<double> b;                                           \
        pop(stack, a, b);                                                 \
        push(stack, op);                                                  \
      },                                                                  \
      aliasAnalysisFromSchema()),                                         \
      OperatorGeneratorArgs(                                              \
          TORCH_SELECTIVE_SCHEMA(                                         \
              #aten_op ".complex_float(complex a, float b) -> " #result), \
          [](Stack& stack) {                                              \
            c10::complex<double> a;                                       \
            double b;                                                     \
            pop(stack, a, b);                                             \
            push(stack, op);                                              \
          },                                                              \
          aliasAnalysisFromSchema())

#define DEFINE_SCALAR_BINARY_OP_WITH_COMPLEX_AVOID_COLLISION_GENERIC( \
    aten_op, int_op, float_op, complex_op, result, string_val)        \
  OperatorGeneratorArgs(                                              \
      TORCH_SELECTIVE_SCHEMA(#aten_op string_val                      \
                             "(Scalar a, Scalar b) -> " #result),     \
      [](Stack& stack) {                                              \
        IValue x, y;                                                  \
        pop(stack, x, y);                                             \
        if (x.isComplexDouble()) {                                    \
          c10::complex<double> a = x.toComplexDouble();               \
          if (y.isComplexDouble()) {                                  \
            c10::complex<double> b = y.toComplexDouble();             \
            push(stack, complex_op);                                  \
          } else if (y.isDouble()) {                                  \
            double b = y.toDouble();                                  \
            push(stack, complex_op);                                  \
          } else {                                                    \
            int64_t b = y.toInt();                                    \
            push(stack, complex_op);                                  \
          }                                                           \
        } else if (x.isDouble()) {                                    \
          double a = x.toDouble();                                    \
          if (y.isComplexDouble()) {                                  \
            c10::complex<double> b = y.toComplexDouble();             \
            push(stack, complex_op);                                  \
          } else if (y.isDouble()) {                                  \
            double b = y.toDouble();                                  \
            push(stack, float_op);                                    \
          } else {                                                    \
            int64_t b = y.toInt();                                    \
            push(stack, float_op);                                    \
          }                                                           \
        } else {                                                      \
          int64_t a = x.toInt();                                      \
          if (y.isComplexDouble()) {                                  \
            c10::complex<double> b = y.toComplexDouble();             \
            push(stack, complex_op);                                  \
          } else if (y.isDouble()) {                                  \
            double b = y.toDouble();                                  \
            push(stack, float_op);                                    \
          } else {                                                    \
            int64_t b = y.toInt();                                    \
            push(stack, int_op);                                      \
          }                                                           \
        }                                                             \
      },                                                              \
      aliasAnalysisFromSchema())

#define DEFINE_SCALAR_BINARY_OP_WITH_COMPLEX_WITHOUT_INT_COMPLEX_PAIR(     \
    aten_op, int_op, float_op, complex_op, result)                         \
  OperatorGeneratorArgs(                                                   \
      TORCH_SELECTIVE_SCHEMA(#aten_op "(Scalar a, Scalar b) -> " #result), \
      [](Stack& stack) {                                                   \
        IValue x, y;                                                       \
        pop(stack, x, y);                                                  \
        if (x.isComplexDouble()) {                                         \
          c10::complex<double> a = x.toComplexDouble();                    \
          if (y.isComplexDouble()) {                                       \
            c10::complex<double> b = y.toComplexDouble();                  \
            push(stack, complex_op);                                       \
          } else if (y.isDouble()) {                                       \
            double b = y.toDouble();                                       \
            push(stack, complex_op);                                       \
          }                                                                \
        } else if (x.isDouble()) {                                         \
          double a = x.toDouble();                                         \
          if (y.isComplexDouble()) {                                       \
            c10::complex<double> b = y.toComplexDouble();                  \
            push(stack, complex_op);                                       \
          } else if (y.isDouble()) {                                       \
            double b = y.toDouble();                                       \
            push(stack, float_op);                                         \
          } else {                                                         \
            int64_t b = y.toInt();                                         \
            push(stack, float_op);                                         \
          }                                                                \
        } else {                                                           \
          int64_t a = x.toInt();                                           \
          if (y.isDouble()) {                                              \
            double b = y.toDouble();                                       \
            push(stack, float_op);                                         \
          } else if (y.isInt()) {                                          \
            int64_t b = y.toInt();                                         \
            push(stack, int_op);                                           \
          }                                                                \
        }                                                                  \
      },                                                                   \
      aliasAnalysisFromSchema())

#define DEFINE_SCALAR_BINARY_OP_WITH_COMPLEX(                   \
    aten_op, int_op, float_op, complex_op, result)              \
  DEFINE_SCALAR_BINARY_OP_WITH_COMPLEX_AVOID_COLLISION_GENERIC( \
      aten_op, int_op, float_op, complex_op, result, "")

#define DEFINE_BINARY_OP_WITH_COMPLEX(aten_op, op)                          \
  DEFINE_GENERIC_OP_WITH_COMPLEX(aten_op, op, op, op, int, float, complex), \
      DEFINE_INT_COMPLEX_OP(aten_op, op, complex),                          \
      DEFINE_FLOAT_COMPLEX_OP(aten_op, op, complex),                        \
      DEFINE_INT_FLOAT_OP(aten_op, op, float),                              \
      DEFINE_SCALAR_BINARY_OP_WITH_COMPLEX(aten_op, op, op, op, Scalar)

#define DEFINE_COMPARISON_OP_WITH_COMPLEX(aten_op, op)                   \
  DEFINE_GENERIC_OP_WITH_COMPLEX(aten_op, op, op, op, bool, bool, bool), \
      DEFINE_INT_FLOAT_OP(aten_op, op, bool),                            \
      DEFINE_FLOAT_COMPLEX_OP(aten_op, op, bool),                        \
      DEFINE_SCALAR_BINARY_OP_WITH_COMPLEX_WITHOUT_INT_COMPLEX_PAIR(     \
          aten_op, op, op, op, bool),                                    \
      DEFINE_STR_CMP_OP(aten_op, op)

} // namespace jit
} // namespace torch
