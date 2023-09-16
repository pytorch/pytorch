#include <ATen/autocast_mode.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/mobile/promoted_prim_ops.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>
#include <torch/csrc/jit/runtime/slice_indices_adjust.h>
#include <torch/library.h>

#include <algorithm>
#include <bitset>
#include <cctype>
#include <cmath>
#include <exception>
#include <fstream>
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

std::string stringSlice(
    std::string string,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step) {
  int64_t start_val = start.has_value() ? start.value() : INT64_MAX;
  int64_t end_val = end.has_value() ? end.value() : INT64_MAX;

  const int64_t num_vals =
      slice_indices_adjust(string.size(), &start_val, &end_val, step);

  int64_t i = start_val;
  std::string result = "";
  for (const auto j : c10::irange(num_vals)) {
    (void)j; // Suppress unused variable warning
    result += string[i];
    i += step;
  }

  return result;
}

// consecutive whitespace are regarded as a single separator,
// the result will contain no empty strings at the start or end
// if the string has leading or trailing whitespace.
c10::List<std::string> splitNoneSeparator(const std::string& string) {
  c10::List<std::string> splits;
  // whitespaces includes tab, space and
  // the delimiters defined in the implementation of splitlines
  std::string whitespaces =
      " \t\n\r\r\n\v\x0b\f\x0c\x1c\x1d\x1e\x85\u2028\u2029";
  std::string::size_type prev_pos = 0;
  std::string::size_type pos = 0;

  while ((pos = string.find_first_of(whitespaces, pos)) != std::string::npos) {
    auto substr = string.substr(prev_pos, pos - prev_pos);
    // skip the whitespaces as the Python split() method
    if (!substr.empty()) {
      splits.emplace_back(substr);
    }
    pos++;
    prev_pos = pos;
  }
  if (prev_pos != string.size()) {
    splits.emplace_back(string.substr(prev_pos));
  }
  return splits;
}

template <typename T, typename U>
auto powWrapper(T a, U b) {
  TORCH_CHECK(
      !(a == 0.0 && b < 0.0), "0.0 cannot be raised to a negative power")
  return pow(a, b);
}

static const std::vector<OperatorGeneratorArgs> opGenArgs{
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::str(t elem) -> str"),
        [](Stack& stack) {
          std::stringstream ss;
          ss << pop(stack);
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::list(str t) -> str[]"),
        [](Stack& stack) {
          auto str = pop(stack).toStringRef();
          c10::List<std::string> chars;
          chars.reserve(str.size());
          for (auto c : str) {
            chars.push_back(std::string(1, c));
          }
          push(stack, std::move(chars));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::cpu(Tensor(a) self) -> Tensor(a|b)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.cpu());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::numpy_T.a(Tensor(a) self) -> Tensor(a)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.numpy_T());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::matrix_H.a(Tensor(a) self) -> Tensor(a)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.matrix_H());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::mT.a(Tensor(a) self) -> Tensor(a)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.mT());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::mH.a(Tensor(a) self) -> Tensor(a)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.mH());
        },
        aliasAnalysisFromSchema()),

    // only used internally in range() translation
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__range_length(int lo, int hi, int step) -> int"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int64_t lo, hi, step;
          pop(stack, lo, hi, step);
          // error handling when step_val = 0 during runtime
          if (step == 0) {
            throw std::runtime_error("range() arg 3 must not be zero");
          }
          if (step > 0 && lo < hi) {
            push(stack, 1 + (hi - 1 - lo) / step);
          } else if (step < 0 && lo > hi) {
            push(stack, 1 + (lo - 1 - hi) / (0 - step));
          } else {
            push(stack, 0);
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__derive_index(int index, int start, int step) -> int"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int64_t index, start, step;
          pop(stack, index, start, step);
          push(stack, start + index * step);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::TupleUnpack(Any tup) -> ..."),
        [](Stack& stack) { tupleUnpack(stack); },
        aliasAnalysisSpecialCase()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::unchecked_cast(t x) -> t"),
        noop,
        aliasAnalysisSpecialCase()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::IntImplicit(Tensor a) -> int"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          checkImplicitTensorToNum(a, /*to int*/ true);
          push(stack, a.item<int64_t>());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ComplexImplicit(Tensor a) -> complex"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          checkImplicitTensorToNum(a, /*to int*/ false);
          push(stack, a.item<c10::complex<double>>());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::FloatImplicit(Tensor a) -> float"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          checkImplicitTensorToNum(a, /*to int*/ false);
          push(stack, a.item<double>());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ScalarImplicit(Tensor a) -> Scalar"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          checkImplicitTensorToNum(a, /*to int*/ false);
          push(stack, a.item());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Bool.Tensor(Tensor a) -> bool"),
        boolTensor,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Bool.int(int a) -> bool"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int64_t i;
          pop(stack, i);
          push(stack, (bool)i);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Bool.float(float a) -> bool"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          double d;
          pop(stack, d);
          push(stack, (bool)d);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Int.Tensor(Tensor a) -> int"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.item<int64_t>());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Int.bool(bool a) -> int"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          bool b;
          pop(stack, b);
          push(stack, static_cast<int64_t>(b));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Int.float(float a) -> int"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          double d;
          pop(stack, d);
          push(stack, static_cast<int64_t>(d));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Int.Scalar(Scalar a) -> int"),
        [](Stack& stack) {
          IValue scalar;
          pop(stack, scalar);
          if (scalar.isInt()) {
            push(stack, std::move(scalar));
          } else {
            // toScalar() needed to avoid strict type check in IValue::toInt.
            push(stack, static_cast<int64_t>(scalar.toScalar().toInt()));
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Int.str(str a) -> int"),
        [](Stack& stack) {
          auto s = pop(stack).toString();
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          std::string::size_type sz;
          int64_t val = static_cast<int64_t>(c10::stoll(s->string(), &sz));
          if (sz == s->string().size()) {
            push(stack, val);
          } else {
            std::stringstream error_str;
            error_str << "invalid literal for int() "
                      << "with base 10: '" << s->string() << "'";
            throw std::runtime_error(error_str.str());
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Float.Tensor(Tensor a) -> float"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.item<double>());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Float.Scalar(Scalar a) -> float"),
        [](Stack& stack) {
          IValue scalar;
          pop(stack, scalar);
          if (scalar.isDouble()) {
            push(stack, std::move(scalar));
          } else if (scalar.isComplexDouble()) {
            push(stack, scalar.toComplexDouble().real());
          } else {
            push(stack, static_cast<double>(scalar.toInt()));
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Float.int(int a) -> float"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int64_t i;
          pop(stack, i);
          push(stack, (float)i);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Float.bool(bool a) -> float"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          bool b;
          pop(stack, b);
          push(stack, (float)b);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Float.str(str a) -> float"),
        [](Stack& stack) {
          auto s = pop(stack).toString();
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          std::string::size_type sz;
          double b = c10::stod(s->string(), &sz);
          if (sz == s->string().size()) {
            push(stack, b);
          } else {
            std::stringstream error_str;
            error_str << "could not convert string "
                      << "to float: '" << s->string() << "'";
            throw std::runtime_error(error_str.str());
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Complex.Scalar(Scalar a) -> complex"),
        [](Stack& stack) {
          IValue scalar;
          pop(stack, scalar);
          if (scalar.isComplexDouble()) {
            push(stack, std::move(scalar));
          } else if (scalar.isDouble()) {
            push(stack, c10::complex<double>(scalar.toDouble(), 0));
          } else {
            push(stack, c10::complex<double>(scalar.toInt(), 0));
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::Complex.Tensor_Tensor(Tensor a, Tensor b) -> complex"),
        [](Stack& stack) {
          at::Tensor a, b;
          pop(stack, a, b);
          push(stack, c10::complex<double>(a.item<double>(), b.item<double>()));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::format(str self, ...) -> str"),
        [](Stack& stack) { aten_format(stack); },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::einsum.sublist(Tensor a, ...) -> Tensor"),
        [](Stack& stack) {
          size_t num_inputs = pop(stack).toInt();
          einsum(stack, num_inputs);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::NumToTensor.Scalar(Scalar a) -> Tensor"),
        numToTensorScalar,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "prim::RaiseException(str msg, str? cls=None) -> ()"),
        raiseException,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Size(int[] sizes) -> int[]"),
        [](Stack& stack) {},
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::size(Tensor self) -> int[]"),
        size,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sym_size(Tensor self) -> SymInt[]"),
        sym_size,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::stride(Tensor self) -> int[]"),
        [](Stack& stack) {
          at::Tensor arg = pop(stack).toTensor();
          push(stack, arg.strides());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sym_stride(Tensor self) -> SymInt[]"),
        sym_stride,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::EnumName(AnyEnumType enum) -> str"),
        [](Stack& stack) {
          IValue e = pop(stack);
          push(stack, e.toEnumHolder()->name());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::EnumValue.int(AnyEnumType enum) -> int"),
        [](Stack& stack) {
          IValue e = pop(stack);
          push(stack, e.toEnumHolder()->value());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "prim::EnumValue.float(AnyEnumType enum) -> float"),
        [](Stack& stack) {
          IValue e = pop(stack);
          push(stack, e.toEnumHolder()->value());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::EnumValue.str(AnyEnumType enum) -> str"),
        [](Stack& stack) {
          IValue e = pop(stack);
          push(stack, e.toEnumHolder()->value());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        // note the compiler knows to type TupleIndex more accurately than it
        // is listed here.
        TORCH_SELECTIVE_SCHEMA("prim::TupleIndex(Any tup, int i) -> Any"),
        tupleIndex,
        aliasAnalysisSpecialCase()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ne.int_list(int[] a, int[] b) -> bool"),
        listNe<int64_t>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "prim::unchecked_unwrap_optional(t(a)? optional) -> t(a)"),
        noop,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::device(Tensor a) -> Device"),
        device,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::dtype(Tensor a) -> int"),
        dtype,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::layout(Tensor a) -> Layout"),
        layout,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::__not__(bool self) -> bool"),
        _not,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::__is__(t1 self, t2 obj) -> bool"),
        is,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::__isnot__(t1 self, t2 obj) -> bool"),
        isNot,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::element_size(Tensor self) -> int"),
        [](Stack& stack) {
          at::Tensor arg = pop(stack).toTensor();
          push(stack, arg.element_size());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::numel(Tensor self) -> int"),
        [](Stack& stack) {
          at::Tensor arg = pop(stack).toTensor();
          push(stack, arg.numel());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::dim(Tensor self) -> int"),
        dim,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::get_device(Tensor self) -> int"),
        [](Stack& stack) {
          RECORD_FUNCTION("get_device", c10::ArrayRef<const c10::IValue>{});
          auto result =
              at::get_device((std::move(peek(stack, 0, 1))).toTensor());
          drop(stack, 1);
          pack(stack, result);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::storage_offset(Tensor self) -> int"),
        [](Stack& stack) {
          RECORD_FUNCTION("storage_offset", c10::ArrayRef<const c10::IValue>{});
          auto result =
              ((std::move(peek(stack, 0, 1))).toTensor()).storage_offset();
          drop(stack, 1);
          pack(stack, result);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::is_contiguous(Tensor self) -> bool"),
        [](Stack& stack) {
          RECORD_FUNCTION("is_contiguous", c10::ArrayRef<const c10::IValue>{});
          auto result =
              ((std::move(peek(stack, 0, 1))).toTensor()).is_contiguous();
          drop(stack, 1);
          pack(stack, result);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::is_contiguous.memory_format(Tensor self, MemoryFormat memory_format) -> bool"),
        [](Stack& stack) {
          auto memory_format = pop(stack).toMemoryFormat();
          auto t = pop(stack).toTensor();
          push(stack, t.is_contiguous(memory_format));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        // NB: intentionally suffixed with extra _format to prevent tests for
        // "_like" suffix from triggering on this
        TORCH_SELECTIVE_SCHEMA(
            "aten::is_strides_like_format(Tensor self, MemoryFormat memory_format) -> bool"),
        [](Stack& stack) {
          auto memory_format = pop(stack).toMemoryFormat();
          auto t = pop(stack).toTensor();
          push(stack, t.unsafeGetTensorImpl()->is_strides_like(memory_format));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::is_non_overlapping_and_dense(Tensor self) -> bool"),
        [](Stack& stack) {
          auto t = pop(stack).toTensor();
          push(stack, t.unsafeGetTensorImpl()->is_non_overlapping_and_dense());
        },
        aliasAnalysisFromSchema()),
    // these ops are generic over the list element type.
    // CREATING GENERIC_LIST_OPS
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::select.t(t[](a) list, int idx) -> t(*)"),
        listSelect,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__getitem__.t(t[](a) list, int idx) -> t(*)"),
        listSelect,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::append.t(t[](a!) self, t(c -> *) el) -> t[](a!)"),
        listAppend,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::reverse.t(t[](a!) self) -> ()"),
        listReverse,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::extend.t(t[](a!) self, t[] other) -> ()"),
        listExtend,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::copy.t(t[](a) self) -> t[]"),
        listCopy,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_set_item.t(t [](a!) l, int idx, t(b -> *) el) -> t[](a!)"),
        listSetItem,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::clear.t(t[](a!) self) -> ()"),
        listClear,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Delete.t(t[](a!) self, int idx) -> ()"),
        listDelete,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::insert.t(t[](a!) self, int idx, t(b -> *) el) -> ()"),
        listInsert,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::pop.t(t[](a!) self, int idx=-1) -> t(*)"),
        listPop,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::add.t(t[] a, t[] b) -> t[]"),
        listAdd,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::add_.t(t[](a!) self, t[] b) -> t[]"),
        listInplaceAdd,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::slice.t(t[] l, int? start=None, int? end=None, int step=1) -> t[]"),
        listSlice,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::list.t(t[] l) -> t[]"),
        listList,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::mul.left_t(t[] l, int n) -> t[]"),
        listMulIntLeft,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::mul.right_(int n, t[] l) -> t[]"),
        listMulIntRight,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::mul_.t(t[](a!) l, int n) -> t[](a!)"),
        listMulIntLeftInPlace,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::len.t(t[] a) -> int"),
        listLen,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::eq.int_list(int[] a, int[] b) -> bool"),
        listEq<int64_t>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::eq.device(Device a, Device b) -> bool"),
        [](Stack& stack) {
          auto a = pop(stack).toDevice();
          auto b = pop(stack).toDevice();
          push(stack, a == b);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ne.device(Device a, Device b) -> bool"),
        [](Stack& stack) {
          auto a = pop(stack).toDevice();
          auto b = pop(stack).toDevice();
          push(stack, a != b);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::eq.bool(bool a, bool b) -> bool"),
        [](Stack& stack) {
          auto a = pop(stack);
          auto b = pop(stack);
          push(stack, a == b);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ne.bool(bool a, bool b) -> bool"),
        [](Stack& stack) {
          auto a = pop(stack);
          auto b = pop(stack);
          push(stack, a != b);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::is_autocast_enabled() -> bool"),
        [](Stack& stack) {
#if defined BUILD_LITE_INTERPRETER || defined C10_MOBILE
          bool enabled = false;
#else
          bool enabled = at::autocast::is_enabled();
#endif
          push(stack, enabled);
        },
        aliasAnalysisConservative()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::is_autocast_cpu_enabled() -> bool"),
        [](Stack& stack) {
#if defined BUILD_LITE_INTERPRETER || defined C10_MOBILE
          bool enabled = false;
#else
          bool enabled = at::autocast::is_cpu_enabled();
#endif
          push(stack, enabled);
        },
        aliasAnalysisConservative()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::Uninitialized() -> Any"),
        unInitialized,
        aliasAnalysisSpecialCase()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::Print(...) -> ()"),
        [](Stack& stack) {
          auto num_inputs = pop(stack).toInt();
          std::stringstream ss;
          bool first = true;
          for (const IValue& i : last(stack, num_inputs)) {
            if (!first)
              ss << " ";
            first = false;
            ss << i;
          }
          drop(stack, num_inputs);
          ss << std::endl;
          auto* handler = getPrintHandler();
          TORCH_INTERNAL_ASSERT(handler);
          handler(ss.str());
        },
        aliasAnalysisSpecialCase()),
    // This is an alternative to aten::cat op that takes variable number of
    // parameters as input.
    // Format:
    //    prim::VarConcat(Tensors..., dim) -> Tensor
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::VarConcat(...) -> Tensor"),
        [](Stack& stack) {
          auto num_inputs = pop(stack).toInt();
          auto dim = pop(stack).toInt();
          std::vector<at::Tensor> inputs(num_inputs - 1);
          for (int i = 0; i < num_inputs - 1; ++i) {
            inputs[num_inputs - 2 - i] = pop(stack).toTensor();
          }
          push(stack, at::cat(inputs, dim));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::VarStack(...) -> Tensor"),
        [](Stack& stack) {
          auto num_inputs = pop(stack).toInt();
          auto dim = pop(stack).toInt();
          std::vector<at::Tensor> inputs(num_inputs - 1);
          for (int i = 0; i < num_inputs - 1; ++i) {
            inputs[num_inputs - 2 - i] = pop(stack).toTensor();
          }
          push(stack, at::stack(inputs, dim));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "prim::IfThenElse(bool cond, Any(a) x, Any(b) y) -> Any(a|b)"),
        [](Stack& stack) {
          const auto cond = stack[stack.size() - 3].toBool();
          stack[stack.size() - 3] =
              std::move(stack[stack.size() - (cond ? 2 : 1)]);
          stack.pop_back();
          stack.pop_back();
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::eq.enum(AnyEnumType a, AnyEnumType b) -> bool"),
        [](Stack& stack) {
          IValue x = pop(stack);
          IValue y = pop(stack);
          push(stack, x == y);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::ne.enum(AnyEnumType a, AnyEnumType b) -> bool"),
        [](Stack& stack) {
          IValue x = pop(stack);
          IValue y = pop(stack);
          push(stack, x != y);
        },
        aliasAnalysisFromSchema()),
    // We define aten::dequantize in both native_functions.yaml and here,
    // however, aten::dequantize.any defined here overrides
    // aten::dequantize.tensors in native_functions.yaml. The variants here
    // are only for graph mode quantization, and they should be removed once
    // we deprecate graph mode quantization, and use the variants in
    // native_functions.yaml.
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::dequantize.tensor(Tensor qtensor) -> Tensor"),
        [](Stack& stack) {
          at::Tensor qtensor;
          pop(stack, qtensor);
          push(stack, at::dequantize(qtensor));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::dequantize.list(Tensor[] qtensors) -> Tensor[]"),
        [](Stack& stack) {
          auto qtensors = pop(stack).toTensorVector();
          push(stack, at::dequantize(qtensors));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::dequantize.any(Any tensors) -> Any"),
        [](Stack& stack) { dequantize(stack); },
        aliasAnalysisFromSchema()),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::log, std::log(a), float, float),
    DEFINE_STRING_OP(aten::add, a + b, str),
    DEFINE_COMPARISON_OP_WITH_COMPLEX(aten::eq, a == b),
    DEFINE_COMPARISON_OP_WITH_COMPLEX(aten::ne, a != b),
    DEFINE_GENERIC_OP(
        aten::polar,
        c10::polar(static_cast<double>(a), static_cast<double>(b)),
        c10::polar(static_cast<double>(a), static_cast<double>(b)),
        complex,
        complex),
    DEFINE_INT_FLOAT_OP(
        aten::polar,
        c10::polar(static_cast<double>(a), static_cast<double>(b)),
        complex),
    DEFINE_SCALAR_BINARY_OP_AVOID_COLLISION(
        aten::polar,
        c10::polar(static_cast<double>(a), static_cast<double>(b)),
        c10::polar(static_cast<double>(a), static_cast<double>(b)),
        Scalar),
    DEFINE_COMPARISON_OP(aten::lt, a < b),
    DEFINE_COMPARISON_OP(aten::gt, a > b),
    DEFINE_COMPARISON_OP(aten::le, a <= b),
    DEFINE_COMPARISON_OP(aten::ge, a >= b),
    DEFINE_BINARY_OP_WITH_COMPLEX(aten::add, a + b),
    DEFINE_BINARY_OP_WITH_COMPLEX(aten::sub, a - b),
    DEFINE_BINARY_OP_WITH_COMPLEX(aten::mul, a* b),
    DEFINE_BOOL_OP(aten::__and__, a&& b),
    DEFINE_BOOL_OP(aten::__or__, a || b),
    DEFINE_BOOL_OP(aten::__xor__, a != b),
    DEFINE_UNARY_OP(aten::round, round_to_even(a), float, float),
    DEFINE_UNARY_OP(aten::floor, floor(a), int, int),
    DEFINE_UNARY_OP(aten::ceil, ceil(a), int, int),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::neg, -a, int, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::exp, std::exp(a), float, float),
    // Pass in two ops for handling int and float separately as % in C++ only
    // works for int The modulus calculation is different between C++ and
    // Python (on negative), we preserve the python behavior as it's more
    // common and match python syntax, hence the conversion.
    DEFINE_GENERIC_OP(
        aten::remainder,
        (b + (a % b)) % b,
        fmod((b + fmod(a, b)), b),
        int,
        float),
    DEFINE_INT_FLOAT_OP(aten::remainder, fmod((b + fmod(a, b)), b), float),
    DEFINE_SCALAR_BINARY_OP(
        aten::remainder,
        (b + (a % b)) % b,
        fmod((b + fmod(a, b)), b),
        Scalar),
    // NB: This is the python truediv operation
    DEFINE_GENERIC_OP_WITH_COMPLEX(
        aten::div,
        static_cast<double>(a) / static_cast<double>(b),
        a / b,
        a / b,
        float,
        float,
        complex),
    DEFINE_SCALAR_BINARY_OP(
        aten::div,
        static_cast<double>(a) / static_cast<double>(b),
        a / b,
        float),
    DEFINE_GENERIC_OP(
        aten::floordiv,
        floordiv(a, b),
        std::floor(a / b),
        int,
        float),
    DEFINE_INT_FLOAT_OP(aten::floordiv, std::floor(a / b), float),
    DEFINE_SCALAR_BINARY_OP(
        aten::floordiv,
        floordiv(a, b),
        std::floor(a / b),
        Scalar),
    // int ** int produces a float, because negative exponents produce float
    // results
    DEFINE_GENERIC_OP_WITH_COMPLEX(
        aten::pow,
        static_cast<double>(powWrapper(a, b)),
        static_cast<double>(powWrapper(a, b)),
        static_cast<c10::complex<double>>(pow(a, b)),
        float,
        float,
        complex),
    DEFINE_INT_FLOAT_OP(
        aten::pow,
        static_cast<double>(powWrapper(a, b)),
        float),
    DEFINE_FLOAT_COMPLEX_OP(aten::pow, pow(a, b), complex),
    DEFINE_SCALAR_BINARY_OP_AVOID_COLLISION(
        aten::pow,
        static_cast<double>(pow(a, b)),
        static_cast<double>(pow(a, b)),
        float),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::pow.int_to_int(int a, int b) -> int"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int64_t a, b;
          pop(stack, a, b);
          push(stack, powWrapper(a, b));
        },
        aliasAnalysisFromSchema()),
    // min and max are in prim:: because there is a difference between
    // the python builtin 'min' and 'torch.min'
    DEFINE_BINARY_OP(prim::min, a < b ? a : b),
    DEFINE_BINARY_OP(prim::max, a > b ? a : b),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::type(Device self) -> str"),
        [](Stack& stack) {
          auto d = pop(stack);
          push(
              stack, DeviceTypeName(d.toDevice().type(), /* lower_case=*/true));
        },
        aliasAnalysisFromSchema()),
    // tensor length op (size of 1st dimension)
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::len.Tensor(Tensor t) -> int"),
        [](Stack& stack) {
          at::Tensor t = pop(stack).toTensor();
          if (t.dim() == 0) {
            AT_ERROR("len() of a 0-d tensor");
          }
          push(stack, t.sizes()[0]);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ord(str string) -> int"),
        [](Stack& stack) {
          auto string = pop(stack).toStringRef();
          TORCH_CHECK(
              string.size() == 1,
              "String for ord() must be 1 character, found ",
              string.size());
          uint8_t ord = string.at(0);
          push(stack, int64_t(ord));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::lower(str self) -> str"),
        [](Stack& stack) {
          auto string = pop(stack).toStringRef();
          std::stringstream ss;
          for (char c : string) {
            ss << static_cast<char>(::tolower(c));
          }
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__contains__.int_list(int[] l, int item) -> bool"),
        listContains<int64_t>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__contains__.str_list(str[] l, str item) -> bool"),
        listContains<std::string>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::len.str(str s) -> int"),
        [](Stack& stack) {
          auto string = pop(stack).toStringRef();
          push(stack, static_cast<int64_t>(string.size()));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::dict() -> Dict(str, Tensor)"),
        [](Stack& stack) {
          auto dict =
              c10::impl::GenericDict(StringType::get(), TensorType::get());
          push(stack, dict);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__getitem__.str(str s, int index) -> str"),
        [](Stack& stack) {
          auto index = pop(stack).toInt();
          auto string = pop(stack).toStringRef();
          auto norm_index = normalizeIndex(index, string.size());
          char c = string.at(norm_index);
          push(stack, std::string(&c, 1));
        },
        aliasAnalysisFromSchema()),
#define CREATE_COPY_OP(other_type, c_type)                               \
  OperatorGeneratorArgs(                                                 \
      TORCH_SELECTIVE_SCHEMA("aten::copy_." #other_type                  \
                             "(Tensor(a!) self, " #other_type            \
                             " other) -> Tensor(a!)"),                   \
      [](Stack& stack) {                                                 \
        at::Tensor t;                                                    \
        c_type other;                                                    \
        pop(stack, t, other);                                            \
        std::move(t) = other; /* NOLINT(bugprone-use-after-move) */      \
        push(stack, std::move(t)); /* NOLINT(bugprone-use-after-move) */ \
      },                                                                 \
      aliasAnalysisFromSchema())

    CREATE_COPY_OP(Tensor, at::Tensor),
    CREATE_COPY_OP(int, int64_t),
    CREATE_COPY_OP(float, double),
#undef CREATE_COPY_OP
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::backward(Tensor self, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False) -> ()"),
        [](Stack& stack) {
          bool create_graph = pop(stack).toBool();
          auto retain_graph = pop(stack).toOptional<bool>();
          IValue gradient_ivalue = pop(stack);
          at::Tensor gradient = gradient_ivalue.isNone()
              ? at::Tensor()
              : gradient_ivalue.toTensor();
          at::Tensor self = pop(stack).toTensor();
          bool keep_graph = retain_graph ? retain_graph.value() : create_graph;
          self.backward(gradient, keep_graph, create_graph);
        },
        aliasAnalysisConservative()),
    //
    // create a clone of these declarations with a _hacked_twin overload name
    // and nullability scrubbed from TensorList arg types
    // TOOD find out why this exists and how to do it without the hack
    //
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::index.Tensor_hacked_twin(Tensor self, Tensor[] indices) -> Tensor"),
        [](Stack& stack) {
          auto indices = pop(stack).to<c10::List<at::Tensor>>();
          c10::List<c10::optional<at::Tensor>> opt_list_indices;
          opt_list_indices.reserve(indices.size());
          for (const auto& ten : indices) {
            opt_list_indices.push_back(ten);
          }
          auto self = pop(stack).toTensor();
          auto result = at::index(self, opt_list_indices);
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_unsafe_index.Tensor_hacked_twin(Tensor self, Tensor[] indices) -> Tensor"),
        [](Stack& stack) {
          auto indices = pop(stack).to<c10::List<at::Tensor>>();
          c10::List<c10::optional<at::Tensor>> opt_list_indices;
          opt_list_indices.reserve(indices.size());
          for (const auto& ten : indices) {
            opt_list_indices.push_back(ten);
          }
          auto self = pop(stack).toTensor();
          auto result = at::_unsafe_index(self, opt_list_indices);
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_index_put_impl_.hacked_twin(Tensor(a!) self, Tensor[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)"),
        [](Stack& stack) {
          auto unsafe = pop(stack).toBool();
          auto accumulate = pop(stack).toBool();
          auto values = pop(stack).toTensor();
          auto indices = pop(stack).to<c10::List<at::Tensor>>();
          c10::List<c10::optional<at::Tensor>> opt_list_indices;
          opt_list_indices.reserve(indices.size());
          for (const auto& ten : indices) {
            opt_list_indices.push_back(ten);
          }
          auto self = pop(stack).toTensor();
          auto result = at::_index_put_impl_(
              self, opt_list_indices, values, accumulate, unsafe);
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::index_put_.hacked_twin(Tensor(a!) self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)"),
        [](Stack& stack) {
          auto accumulate = pop(stack).toBool();
          auto values = pop(stack).toTensor();
          auto indices = pop(stack).to<c10::List<at::Tensor>>();
          c10::List<c10::optional<at::Tensor>> opt_list_indices;
          opt_list_indices.reserve(indices.size());
          for (const auto& ten : indices) {
            opt_list_indices.push_back(ten);
          }
          auto self = pop(stack).toTensor();
          auto result =
              at::index_put_(self, opt_list_indices, values, accumulate);
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::index_put.hacked_twin(Tensor self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor"),
        [](Stack& stack) {
          auto accumulate = pop(stack).toBool();
          auto values = pop(stack).toTensor();
          auto indices = pop(stack).to<c10::List<at::Tensor>>();
          c10::List<c10::optional<at::Tensor>> opt_list_indices;
          opt_list_indices.reserve(indices.size());
          for (const auto& ten : indices) {
            opt_list_indices.push_back(ten);
          }
          auto self = pop(stack).toTensor();
          auto result =
              at::index_put(self, opt_list_indices, values, accumulate);
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_unsafe_index_put.hacked_twin(Tensor self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor"),
        [](Stack& stack) {
          auto accumulate = pop(stack).toBool();
          auto values = pop(stack).toTensor();
          auto indices = pop(stack).to<c10::List<at::Tensor>>();
          c10::List<c10::optional<at::Tensor>> opt_list_indices;
          opt_list_indices.reserve(indices.size());
          for (const auto& ten : indices) {
            opt_list_indices.push_back(ten);
          }
          auto self = pop(stack).toTensor();
          auto result =
              at::_unsafe_index_put(self, opt_list_indices, values, accumulate);
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    // reference function parse_to_conversion in python_arg_parsing.h
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::to.prim_Device(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          bool non_blocking;
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          bool copy;
          pop(stack, non_blocking, copy);
          c10::optional<at::ScalarType> scalarType =
              pop(stack).toOptional<at::ScalarType>();
          c10::optional<c10::Device> device =
              pop(stack).toOptional<c10::Device>();
          at::Tensor self = pop(stack).toTensor();
          push(
              stack, to_dispatch(self, device, scalarType, non_blocking, copy));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::to.prim_dtype(Tensor(a) self, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)"),
        toPrimDType,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_cuda(Tensor a) -> bool"),
        isCuda,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_cpu(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_cpu());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_xla(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_xla());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_mtia(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_mtia());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_xpu(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_xpu());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::data(Tensor(a) a) -> Tensor(a)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, autograd::Variable(a).variable_data());
        },
        aliasAnalysisFromSchema()),
// these ops are not defined for Tensor
#define CREATE_COMPARATOR_LIST_OPS_SPECIALIZED(decl_type, value_type)        \
  OperatorGeneratorArgs(                                                     \
      TORCH_SELECTIVE_SCHEMA("prim::min." decl_type "_list(" decl_type       \
                             "[] l, " decl_type "[] r) -> " decl_type "[]"), \
      minList<value_type>,                                                   \
      aliasAnalysisFromSchema()),                                            \
      OperatorGeneratorArgs(                                                 \
          TORCH_SELECTIVE_SCHEMA("prim::max." decl_type "_list(" decl_type   \
                                 "[] l, " decl_type "[] r) -> " decl_type    \
                                 "[]"),                                      \
          maxList<value_type>,                                               \
          aliasAnalysisFromSchema()),                                        \
      OperatorGeneratorArgs(                                                 \
          TORCH_SELECTIVE_SCHEMA("prim::min.self_" decl_type "(" decl_type   \
                                 "[] self) -> " decl_type),                  \
          listMin<value_type>,                                               \
          aliasAnalysisFromSchema()),                                        \
      OperatorGeneratorArgs(                                                 \
          TORCH_SELECTIVE_SCHEMA("prim::max.self_" decl_type "(" decl_type   \
                                 "[] self) -> " decl_type),                  \
          listMax<value_type>,                                               \
          aliasAnalysisFromSchema()),
    CREATE_COMPARATOR_LIST_OPS_SPECIALIZED("int", int64_t)
        CREATE_COMPARATOR_LIST_OPS_SPECIALIZED("float", double)
            CREATE_COMPARATOR_LIST_OPS_SPECIALIZED("bool", bool)
#undef CREATE_COMPARATOR_LIST_OPS_SPECIALIZED
// python string is methods return false if empty
#define DEFINE_STRING_IS_OP(op_name, char_op)                          \
  OperatorGeneratorArgs(                                               \
      TORCH_SELECTIVE_SCHEMA(#op_name "(str self) -> bool"),           \
      [](Stack& stack) {                                               \
        auto string = pop(stack).toStringRef();                        \
        push(                                                          \
            stack,                                                     \
            string.size() != 0 &&                                      \
                std::all_of(string.begin(), string.end(), [](char c) { \
                  return char_op(c);                                   \
                }));                                                   \
      },                                                               \
      aliasAnalysisFromSchema())

                DEFINE_STRING_IS_OP(aten::isdigit, ::isdigit),
    DEFINE_STRING_IS_OP(aten::isspace, ::isspace),
    DEFINE_STRING_IS_OP(aten::isalnum, ::isalnum),
    DEFINE_STRING_IS_OP(aten::isalpha, ::isalpha),
    DEFINE_STRING_IS_OP(aten::isdecimal, ::isdigit),
    DEFINE_STRING_IS_OP(aten::isnumeric, ::isdigit),

#define DEFINE_STRING_CHAR_MAP_OP(op_name, char_op)         \
  OperatorGeneratorArgs(                                    \
      TORCH_SELECTIVE_SCHEMA(#op_name "(str self) -> str"), \
      [](Stack& stack) {                                    \
        auto string = pop(stack).toStringRef();             \
        std::stringstream ss;                               \
        for (char c : string) {                             \
          ss << static_cast<char>(char_op(c));              \
        }                                                   \
        push(stack, ss.str());                              \
      },                                                    \
      aliasAnalysisFromSchema())

    DEFINE_STRING_CHAR_MAP_OP(aten::upper, ::toupper),
    DEFINE_STRING_CHAR_MAP_OP(aten::swapcase, ([](char c) {
                                if (c == static_cast<char>(::toupper(c))) {
                                  return static_cast<char>(::tolower(c));
                                } else {
                                  return static_cast<char>(::toupper(c));
                                }
                              }))};

static std::vector<c10::optional<Operator>> createOperators(
    const std::vector<OperatorGeneratorArgs>& args) {
  std::vector<c10::optional<Operator>> result;
  result.reserve(args.size());
  for (const auto& arg : args) {
    if (arg.schema_str) {
      if (arg.isOperationCreator) {
        result.push_back(OperatorGenerator(
            arg.schema_str, arg.operationCreator, arg.aliasAnalysis));
      } else {
        result.push_back(OperatorGenerator(
            arg.schema_str, arg.operation, arg.aliasAnalysis));
      }
    }
  }
  return result;
}

RegisterOperators reg(([]() {
  auto v = createOperators(opGenArgs);
  v.emplace_back(Operator(
      prim::tolist,
      // This operator has to be unschematized because the return type
      // depends on the type hint and input. The implementation of this
      // operator below is intended to be as close to the Python
      // implementation in torch/csrc/utils/tensor_list.cpp as possible.
      [](const Node* /*node*/) -> Operation { return toList; },
      aliasAnalysisSpecialCase()));
  return v;
})());

void dictSetItem(Stack& stack) {
  auto value = pop(stack);
  auto idx = pop(stack);
  auto dict = pop(stack).toGenericDict();
  dict.insert_or_assign(std::move(idx), std::move(value));
}

void dictLen(Stack& stack) {
  auto dict = pop(stack).toGenericDict();
  push(stack, int64_t(dict.size()));
}

void dictValues(Stack& stack) {
  auto dict = pop(stack).toGenericDict();
  auto values = c10::impl::GenericList(dict.valueType());
  for (const auto& entry : dict) {
    values.emplace_back(entry.value());
  }
  push(stack, values);
}

void dictKeys(Stack& stack) {
  auto dict = pop(stack).toGenericDict();
  auto keys = c10::impl::GenericList(dict.keyType());
  for (const auto& entry : dict) {
    keys.emplace_back(entry.key());
  }
  push(stack, keys);
}

template <bool has_default>
void dictGet(Stack& stack) {
  IValue default_value;
  if (has_default) {
    default_value = pop(stack);
  }
  auto key = pop(stack);
  auto dict = pop(stack).toGenericDict();
  auto value = dict.find(key);
  if (value == dict.end()) {
    push(stack, std::move(default_value));
  } else {
    push(stack, value->value());
  }
}

// If the key is in the dict, return it. Else set it to the default value and
// return that.
void dictSetDefault(Stack& stack) {
  auto default_value = pop(stack);
  auto key = pop(stack);
  auto dict = pop(stack).toGenericDict();
  auto value = dict.find(key);
  if (value == dict.end()) {
    dict.insert(key, default_value);
    push(stack, std::move(default_value));
  } else {
    push(stack, value->value());
  }
}

template <bool has_default>
void dictPop(Stack& stack) {
  IValue default_value;
  if (has_default) {
    default_value = pop(stack);
  }
  auto key = pop(stack);
  auto dict = pop(stack).toGenericDict();
  auto iter = dict.find(key);
  if (iter == dict.end()) {
    if (has_default) {
      push(stack, default_value);
    } else {
      AT_ERROR("KeyError: ", key);
    }
  } else {
    // note: before erase
    push(stack, iter->value());
    auto erase_count = dict.erase(key);
    TORCH_CHECK(
        erase_count == 1, "Expected to erase 1 item, found ", erase_count);
  }
}

void dictDelete(Stack& stack) {
  dictPop<false>(stack);
  // pop pushes an item on the stack but delete does not, so get rid of it
  pop(stack);
}

void dictPopItem(Stack& stack) {
  auto dict = pop(stack).toGenericDict();
  if (dict.empty()) {
    AT_ERROR("popitem(): dictionary is empty");
  }
  auto head_item = dict.begin();

  IValue tuple =
      c10::ivalue::Tuple::create({head_item->key(), head_item->value()});
  auto erase_count = dict.erase(head_item->key());
  TORCH_CHECK(
      erase_count == 1, "Expected to erase 1 item, found ", erase_count);
  push(stack, tuple);
}

void dictContains(Stack& stack) {
  auto key = pop(stack);
  auto dict = pop(stack).toGenericDict();
  push(stack, dict.contains(key));
}

void dictClear(Stack& stack) {
  auto dict = pop(stack).toGenericDict();
  dict.clear();
}

void dictUpdate(Stack& stack) {
  auto to_add = pop(stack).toGenericDict();
  auto dict = pop(stack).toGenericDict();

  for (const auto& item : to_add) {
    dict.insert_or_assign(item.key(), item.value());
  }
}

void dictItems(Stack& stack) {
  auto dict = pop(stack).toGenericDict();
  auto key_type = dict.keyType();
  auto value_type = dict.valueType();
  auto items =
      c10::impl::GenericList(TupleType::create({key_type, value_type}));
  items.reserve(dict.size());
  for (const auto& item : dict) {
    items.emplace_back(c10::ivalue::Tuple::create({item.key(), item.value()}));
  }
  push(stack, std::move(items));
}

void dictCopy(Stack& stack) {
  push(stack, pop(stack).toGenericDict().copy());
}

void dictConstructFromList(Stack& stack) {
  auto input_list = pop(stack);
  auto list = input_list.toList();
  auto tup_type = list.elementType()->expect<TupleType>();
  auto dict = c10::impl::GenericDict(
      tup_type->elements().at(0), tup_type->elements().at(1));
  dict.reserve(list.size());
  for (IValue input : list) {
    const auto& tup = input.toTupleRef().elements();
    dict.insert_or_assign(tup[0], tup[1]);
  }
  push(stack, dict);
}

#define CREATE_DICT_OPS(key_type)                                              \
  OperatorGeneratorArgs(                                                       \
      TORCH_SELECTIVE_SCHEMA("aten::len.Dict_" key_type "(Dict(" key_type      \
                             ", t) self) -> int"),                             \
      dictLen,                                                                 \
      aliasAnalysisFromSchema()),                                              \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA("aten::keys." key_type "(Dict(" key_type      \
                                 ", t) self) -> " key_type "[](*)"),           \
          dictKeys,                                                            \
          aliasAnalysisFromSchema()),                                          \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA("aten::values." key_type "(Dict(" key_type    \
                                 ", t) self) -> t[](*)"),                      \
          dictValues,                                                          \
          aliasAnalysisFromSchema()),                                          \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA("aten::__getitem__.Dict_" key_type            \
                                 "(Dict(" key_type ", t) self, " key_type      \
                                 " key) -> t(*)"),                             \
          dictIndex,                                                           \
          aliasAnalysisFromSchema()),                                          \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA("aten::get." key_type "(Dict(" key_type       \
                                 ", t) self, " key_type " key) -> t(*)?"),     \
          dictGet<false>,                                                      \
          aliasAnalysisFromSchema()),                                          \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA("aten::get.default_" key_type                 \
                                 "(Dict(" key_type ", t) self, " key_type      \
                                 " key, t default_value) -> t(*)"),            \
          dictGet<true>,                                                       \
          aliasAnalysisFromSchema()),                                          \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA(                                              \
              "aten::setdefault." key_type "(Dict(" key_type                   \
              ", t)(a!) self, " key_type                                       \
              "(b -> *) key, t(c -> *) default_value) -> t(*)"),               \
          dictSetDefault,                                                      \
          aliasAnalysisFromSchema()),                                          \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA("aten::Delete.Dict_" key_type                 \
                                 "(Dict(" key_type ", t)(a!) self, " key_type  \
                                 " key) -> ()"),                               \
          dictDelete,                                                          \
          aliasAnalysisFromSchema()),                                          \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA("aten::pop.Dict_" key_type "(Dict(" key_type  \
                                 ", t)(a!) self, " key_type " key) -> t(*)"),  \
          dictPop<false>,                                                      \
          aliasAnalysisFromSchema()),                                          \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA("aten::pop.Dict_default_" key_type            \
                                 "(Dict(" key_type ", t)(a!) self, " key_type  \
                                 " key, t default_value) -> t(*)"),            \
          dictPop<true>,                                                       \
          aliasAnalysisFromSchema()),                                          \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA("aten::popitem." key_type "(Dict(" key_type   \
                                 ", t)(a!) self) -> ((" key_type ", t))"),     \
          dictPopItem,                                                         \
          aliasAnalysisFromSchema()),                                          \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA("aten::clear." key_type "(Dict(" key_type     \
                                 ", t)(a!) self) -> ()"),                      \
          dictClear,                                                           \
          aliasAnalysisFromSchema()),                                          \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA("aten::update." key_type "(Dict(" key_type    \
                                 ", t)(a!) self, Dict(" key_type               \
                                 ", t)(a!) to_add) -> ()"),                    \
          dictUpdate,                                                          \
          aliasAnalysisFromSchema()),                                          \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA("aten::items." key_type "(Dict(" key_type     \
                                 ", t) self) -> ((" key_type ", t)[])"),       \
          dictItems,                                                           \
          aliasAnalysisFromSchema()),                                          \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA("aten::copy.Dict_" key_type "(Dict(" key_type \
                                 ", t)(a) self) -> Dict(" key_type ", t)"),    \
          dictCopy,                                                            \
          aliasAnalysisFromSchema()),                                          \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA("aten::__contains__." key_type                \
                                 "(Dict(" key_type ", t) dict, " key_type      \
                                 " key) -> bool"),                             \
          dictContains,                                                        \
          aliasAnalysisFromSchema()),                                          \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA("aten::_set_item." key_type "(Dict(" key_type \
                                 ", t)(a!) l, " key_type                       \
                                 "(b -> *) idx, t(c -> *) v) -> ()"),          \
          dictSetItem,                                                         \
          aliasAnalysisFromSchema()),                                          \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA("aten::dict." key_type "((" key_type          \
                                 ", tVal)[] inputs) -> Dict(" key_type         \
                                 ", tVal)"),                                   \
          dictConstructFromList,                                               \
          aliasAnalysisFromSchema()),                                          \
      OperatorGeneratorArgs(                                                   \
          TORCH_SELECTIVE_SCHEMA("aten::dict.Dict_" key_type "(Dict(" key_type \
                                 ", t)(a) self) -> Dict(" key_type ", t)"),    \
          dictCopy,                                                            \
          aliasAnalysisFromSchema())

static const std::vector<OperatorGeneratorArgs> dict_ops{
    CREATE_DICT_OPS("str"),
    CREATE_DICT_OPS("int"),
    CREATE_DICT_OPS("bool"),
    CREATE_DICT_OPS("float"),
    CREATE_DICT_OPS("complex"),
    CREATE_DICT_OPS("Tensor"),
};
RegisterOperators reg_dict_ops(createOperators(dict_ops));

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
constexpr c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

// Convert an python index (which may be negative) into an index usable for a
// C++ container
// NOLINTNEXTLINE(clang-diagnostic-unused-function)
int64_t normalizeIndex(int64_t idx, int64_t list_size) {
  if (idx < 0) {
    // Handle negative indexing
    idx = list_size + idx;
  }
  return idx;
}

int64_t stringFindImpl(
    std::string string,
    const std::string& substr,
    int64_t start,
    int64_t end,
    bool reverse = false) {
  int64_t size = string.size();
  if (start < 0) {
    start = std::max(int64_t(0), int64_t(size + start));
  }
  if (end < 0) {
    end = std::max(int64_t(0), int64_t(size + end + 1));
  }
  if (end > start) {
    string = string.substr(start, end - start);
  } else {
    string = "";
  }

  int64_t result = -1;
  if (string.size() >= substr.size()) {
    auto pos = string.find(substr, 0);
    if (reverse) {
      auto rpos = pos;
      do {
        pos = rpos;
        rpos = string.find(substr, pos + 1);
      } while (rpos != std::string::npos);
    }
    if (pos != std::string::npos) {
      result = pos + start;
    }
  }
  return result;
}

// String Ops
// Implementations located in torch/csrc/jit/runtime/register_prim_ops.cpp
static const std::vector<OperatorGeneratorArgs> stringOpGenArgs{
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::slice.str(str string, int? start=None, int? end=None, int step=1) -> str"),
        [](Stack& stack) {
          int64_t step = pop(stack).toInt();
          c10::optional<int64_t> end = pop(stack).toOptional<int64_t>();
          c10::optional<int64_t> start = pop(stack).toOptional<int64_t>();
          std::string string = pop(stack).toStringRef();
          push(stack, stringSlice(string, start, end, step));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::strip(str self, str chars=' \\n\\t\\f\\v') -> str"),
        [](Stack& stack) {
          std::string chars = pop(stack).toStringRef();
          std::string string = pop(stack).toStringRef();
          auto rindex = string.find_last_not_of(chars);
          if (rindex != std::string::npos) {
            string = string.substr(0, rindex + 1);
          } else {
            string = "";
          }
          auto lindex = string.find_first_not_of(chars);
          if (lindex != std::string::npos) {
            string = string.substr(lindex, string.size());
          } else {
            string = "";
          }
          push(stack, string);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::split.str(str self, str? separator=None, int max=-1) -> str[]"),
        [](Stack& stack) {
          int64_t max = pop(stack).toInt();
          IValue ivalue = pop(stack);
          std::string string = pop(stack).toStringRef();

          std::string::size_type prev_pos = 0;
          std::string::size_type pos = 0;
          c10::List<std::string> splits;
          if (ivalue == c10::nullopt) {
            // if separator is not specified,
            // a different splitting algorithm is applied as Python
            splits = splitNoneSeparator(string);
            push(stack, std::move(splits));
            return;
          }

          const std::string& separator = ivalue.toStringRef();

          if (separator.empty()) {
            throw std::runtime_error("ValueError: empty separator");
          }

          auto count = 0;

          while ((pos = string.find(separator, pos)) != std::string::npos) {
            count++;
            if (max >= 0 && count > max) {
              break;
            } else {
              splits.emplace_back(string.substr(prev_pos, pos - prev_pos));
            }
            pos += separator.size();
            prev_pos = pos;
          }
          splits.emplace_back(
              string.substr(prev_pos, string.size() - prev_pos));
          push(stack, std::move(splits));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::splitlines(str self, bool keepends=False) -> str[]"),
        [](Stack& stack) {
          bool keepends = pop(stack).toBool();
          std::string string = pop(stack).toStringRef();
          std::string delimiters =
              "\n\r\r\n\v\x0b\f\x0c\x1c\x1d\x1e\x85\u2028\u2029";
          c10::List<std::string> splits;

          std::string::size_type prev_pos = 0;
          std::string::size_type pos = 0;
          while ((pos = string.find_first_of(delimiters, pos)) !=
                 std::string::npos) {
            splits.emplace_back(string.substr(prev_pos, pos - prev_pos));
            if (keepends) {
              splits.emplace_back(string.substr(pos, 1));
            }
            pos++;
            prev_pos = pos;
          }
          if (prev_pos != string.size()) {
            splits.emplace_back(
                string.substr(prev_pos, string.size() - prev_pos));
          }

          push(stack, std::move(splits));
        },
        aliasAnalysisFromSchema()),
    // upper and lower require there to be at least one alpha character,
    // and ignore all other characters
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::isupper(str self) -> bool"),
        [](Stack& stack) {
          std::string string = pop(stack).toStringRef();
          bool found_alpha = false;
          bool is_upper = true;
          for (size_t i = 0; i < string.size() && is_upper; ++i) {
            char c = string[i];
            found_alpha |= static_cast<bool>(::isalpha(c));
            is_upper &= (!::isalpha(c) || ::isupper(c));
          }
          push(stack, found_alpha && is_upper);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::islower(str self) -> bool"),
        [](Stack& stack) {
          std::string string = pop(stack).toStringRef();
          bool found_alpha = false;
          bool is_lower = true;
          for (size_t i = 0; i < string.size() && is_lower; ++i) {
            char c = string[i];
            found_alpha |= static_cast<bool>(::isalpha(c));
            is_lower &= (!::isalpha(c) || ::islower(c));
          }
          push(stack, found_alpha && is_lower);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::capitalize(str self) -> str"),
        [](Stack& stack) {
          std::string string = pop(stack).toStringRef();
          std::stringstream ss;
          auto first_char = true;
          for (char c : string) {
            if (first_char) {
              ss << static_cast<char>(::toupper(c));
              first_char = false;
            } else {
              ss << static_cast<char>(::tolower(c));
            }
          }
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::title(str self) -> str"),
        [](Stack& stack) {
          std::string string = pop(stack).toStringRef();
          std::stringstream ss;
          bool prev_is_nonalpha = true;
          for (char c : string) {
            if (prev_is_nonalpha) {
              ss << static_cast<char>(::toupper(c));
            } else {
              ss << static_cast<char>(::tolower(c));
            }
            if (::isalpha(c)) {
              prev_is_nonalpha = false;
            } else {
              prev_is_nonalpha = true;
            }
          }
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::center(str self, int width, str fillchar=' ') -> str"),
        [](Stack& stack) {
          std::string fillchar = pop(stack).toStringRef();
          int64_t width = pop(stack).toInt();
          std::string string = pop(stack).toStringRef();
          if (fillchar.size() != 1) {
            // TODO: this should be a TypeError
            throw std::runtime_error(
                "TypeError: The fill character must be exactly one character long");
          }
          if (string.size() > static_cast<std::string::size_type>(width)) {
            push(stack, string);
            return;
          }
          std::stringstream ss;
          std::string::size_type full_padding = width - string.size();
          std::string::size_type l_pad = full_padding / 2;
          std::string::size_type r_pad = (full_padding + 1) / 2;
          if (width % 2) {
            auto tmp = r_pad;
            r_pad = l_pad;
            l_pad = tmp;
          }
          for (std::string::size_type i = 0; i < l_pad; ++i) {
            ss << fillchar;
          }
          ss << string;
          for (std::string::size_type i = 0; i < r_pad; ++i) {
            ss << fillchar;
          }
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),

    // Adapted from
    // https://stackoverflow.com/questions/22489073/counting-the-number-of-occurrences-of-a-string-within-a-string
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::count(str self, str substr, int start=0, int end=-1) -> int"),
        [](Stack& stack) {
          int64_t end = pop(stack).toInt();
          int64_t start = pop(stack).toInt();
          std::string substr = pop(stack).toStringRef();
          std::string string = pop(stack).toStringRef();
          int64_t size = string.size();
          if (start > size) {
            push(stack, 0);
            return;
          }
          if (start < 0) {
            start = std::max(int64_t(0), int64_t(size + start));
          }
          if (end < 0) {
            end = std::max(int64_t(0), int64_t(size + end + 1));
          }

          int64_t occurrences = 0;
          std::string::size_type pos = start;
          while ((pos = string.find(substr, pos)) != std::string::npos) {
            if (pos < static_cast<std::string::size_type>(end)) {
              ++occurrences;
            } else {
              break;
            }
            pos += substr.length();
          }
          push(stack, occurrences);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::endswith(str self, str substr, int start=0, int end=-1) -> bool"),
        [](Stack& stack) {
          int64_t end = pop(stack).toInt();
          int64_t start = pop(stack).toInt();
          std::string substr = pop(stack).toStringRef();
          std::string string = pop(stack).toStringRef();
          int64_t size = string.size();
          if (start < 0) {
            start = std::max(int64_t(0), int64_t(size + start));
          }
          if (end < 0) {
            end = std::max(int64_t(0), int64_t(size + end + 1));
          }

          string = string.substr(start, end - start);

          auto result = false;
          if (string.length() >= substr.length()) {
            result = !string.compare(
                string.length() - substr.length(), substr.length(), substr);
          }
          push(stack, result);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::startswith(str self, str substr, int start=0, int end=-1) -> bool"),
        [](Stack& stack) {
          int64_t end = pop(stack).toInt();
          int64_t start = pop(stack).toInt();
          std::string substr = pop(stack).toStringRef();
          std::string string = pop(stack).toStringRef();
          int64_t size = string.size();
          if (start < 0) {
            start = std::max(int64_t(0), int64_t(size + start));
          }
          if (end < 0) {
            end = std::max(int64_t(0), int64_t(size + end + 1));
          }

          string = string.substr(start, end - start);

          auto result = false;
          if (string.length() >= substr.length()) {
            result = !string.compare(0, substr.length(), substr);
          }
          push(stack, result);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::expandtabs(str self, int tabsize=8) -> str"),
        [](Stack& stack) {
          int64_t tabsize = pop(stack).toInt();
          std::string string = pop(stack).toStringRef();
          std::stringstream ss;
          size_t index = 0;
          for (const auto& c : string) {
            if (c != '\t') {
              ss << c;
              index++;
            } else {
              if (tabsize <= 0) {
                continue;
              }
              do {
                ss << ' ';
                index++;
              } while (index % tabsize);
            }
          }
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::find(str self, str substr, int start=0, int end=-1) -> int"),
        [](Stack& stack) {
          int64_t end = pop(stack).toInt();
          int64_t start = pop(stack).toInt();
          std::string substr = pop(stack).toStringRef();
          std::string string = pop(stack).toStringRef();

          push(stack, stringFindImpl(string, substr, start, end));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::rfind(str self, str substr, int start=0, int end=-1) -> int"),
        [](Stack& stack) {
          int64_t end = pop(stack).toInt();
          int64_t start = pop(stack).toInt();
          std::string substr = pop(stack).toStringRef();
          std::string string = pop(stack).toStringRef();

          push(stack, stringFindImpl(string, substr, start, end, true));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::index.str(str self, str substr, int start=0, int end=-1) -> int"),
        [](Stack& stack) {
          int64_t end = pop(stack).toInt();
          int64_t start = pop(stack).toInt();
          std::string substr = pop(stack).toStringRef();
          std::string string = pop(stack).toStringRef();
          auto result = stringFindImpl(string, substr, start, end);
          if (result < 0) {
            throw std::runtime_error("ValueError: substring not found");
          }
          push(stack, result);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::rindex(str self, str substr, int start=0, int end=-1) -> int"),
        [](Stack& stack) {
          int64_t end = pop(stack).toInt();
          int64_t start = pop(stack).toInt();
          std::string substr = pop(stack).toStringRef();
          std::string string = pop(stack).toStringRef();
          auto result = stringFindImpl(string, substr, start, end, true);
          if (result < 0) {
            throw std::runtime_error("ValueError: substring not found");
          }
          push(stack, result);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::isidentifier(str self) -> bool"),
        [](Stack& stack) {
          std::string string = pop(stack).toStringRef();
          LOG(WARNING)
              << "The isidentifier() implementation being used is from Python 2\n";
          if (string.empty()) {
            push(stack, false);
            return;
          }
          if (::isdigit(string[0])) {
            push(stack, false);
            return;
          }
          auto result = std::all_of(string.begin(), string.end(), [](char c) {
            return ::isalnum(c);
          });
          push(stack, result);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::istitle(str self) -> bool"),
        [](Stack& stack) {
          std::string string = pop(stack).toStringRef();
          auto result = false;

          bool prev_is_alpha = false;
          for (char c : string) {
            if (prev_is_alpha) {
              if (c != static_cast<char>(::tolower(c))) {
                result = false;
                break;
              }
            } else {
              if (c != static_cast<char>(::toupper(c))) {
                result = false;
                break;
              }
              // Only true if there exists at least one alpha
              if (::isalpha(c)) {
                result = true;
              }
            }
            if (::isalpha(c)) {
              prev_is_alpha = true;
            } else {
              prev_is_alpha = false;
            }
          }
          push(stack, result);
        },
        aliasAnalysisFromSchema()),
    // Can't reuse DEFINE_STRING_IS_OP because "" is printable
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::isprintable(str self) -> bool"),
        [](Stack& stack) {
          std::string string = pop(stack).toStringRef();
          auto result = std::all_of(string.begin(), string.end(), [](char c) {
            return ::isalnum(c) || ::ispunct(c) || c == ' ';
          });
          push(stack, result);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::ljust(str self, int width, str fillchar=' ') -> str"),
        [](Stack& stack) {
          std::string fillchar = pop(stack).toStringRef();
          int64_t width = pop(stack).toInt();
          std::string string = pop(stack).toStringRef();
          if (fillchar.size() != 1) {
            // TODO: this should be a TypeError
            throw std::runtime_error(
                "TypeError: The fill character must be exactly one character long");
          }
          auto to_append =
              std::max(int64_t(0), width - static_cast<int64_t>(string.size()));

          std::stringstream ss;
          ss << string;
          for (const auto i : c10::irange(to_append)) {
            (void)i; // Suppress unused variable warning
            ss << fillchar;
          }
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::rjust(str self, int width, str fillchar=' ') -> str"),
        [](Stack& stack) {
          std::string fillchar = pop(stack).toStringRef();
          int64_t width = pop(stack).toInt();
          std::string string = pop(stack).toStringRef();
          if (fillchar.size() != 1) {
            // TODO: this should be a TypeError
            throw std::runtime_error(
                "TypeError: The fill character must be exactly one character long");
          }
          auto to_append =
              std::max(int64_t(0), width - static_cast<int64_t>(string.size()));

          std::stringstream ss;
          for (const auto i : c10::irange(to_append)) {
            (void)i; // Suppress unused variable warning
            ss << fillchar;
          }
          ss << string;
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::zfill(str self, int width) -> str"),
        [](Stack& stack) {
          int64_t width = pop(stack).toInt();
          std::string string = pop(stack).toStringRef();
          auto to_append =
              std::max(int64_t(0), width - static_cast<int64_t>(string.size()));

          std::stringstream ss;
          for (const auto i : c10::irange(to_append)) {
            (void)i; // Suppress unused variable warning
            ss << '0';
          }
          ss << string;
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::lstrip(str self, str chars=' \\n\\t\\f\\v') -> str"),
        [](Stack& stack) {
          std::string chars = pop(stack).toStringRef();
          std::string string = pop(stack).toStringRef();
          auto index = string.find_first_not_of(chars);
          if (index != std::string::npos) {
            string = string.substr(index, string.size());
          } else {
            string = "";
          }
          push(stack, string);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::rstrip(str self, str chars=' \\n\\t\\f\\v') -> str"),
        [](Stack& stack) {
          std::string chars = pop(stack).toStringRef();
          std::string string = pop(stack).toStringRef();
          auto index = string.find_last_not_of(chars);
          if (index != std::string::npos) {
            string = string.substr(0, index + 1);
          } else {
            string = "";
          }
          push(stack, string);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::replace(str self, str old, str new, int max=-1) -> str"),
        [](Stack& stack) {
          int64_t max = pop(stack).toInt();
          std::string new_str = pop(stack).toStringRef();
          std::string old_str = pop(stack).toStringRef();
          std::string string = pop(stack).toStringRef();
          int64_t occurrences = 0;
          std::string::size_type pos = 0;
          while ((pos = string.find(old_str, pos)) != std::string::npos) {
            if (max >= 0 && ++occurrences > max) {
              break;
            }
            string = string.replace(pos, old_str.length(), new_str);
            pos += new_str.length();
          }

          push(stack, string);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::partition(str self, str separator) -> (str, str, str)"),
        [](Stack& stack) {
          std::string separator = pop(stack).toStringRef();
          std::string string = pop(stack).toStringRef();
          auto pos = string.find(separator, 0);
          if (pos == std::string::npos) {
            pos = string.size();
            separator = "";
          }
          auto pre_partition = string.substr(0, pos);
          auto post_partition =
              string.substr(pos + separator.size(), string.size());
          push(stack, pre_partition, separator, post_partition);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::rpartition(str self, str separator) -> (str, str, str)"),
        [](Stack& stack) {
          std::string separator = pop(stack).toStringRef();
          std::string string = pop(stack).toStringRef();
          auto pos = string.find(separator, 0);
          auto rpos = pos;
          do {
            pos = rpos;
            rpos = string.find(separator, pos + 1);
          } while (rpos != std::string::npos);

          if (pos == std::string::npos) {
            pos = 0;
            separator = "";
          }

          auto pre_partition = string.substr(0, pos);
          auto post_partition =
              string.substr(pos + separator.size(), string.size());
          push(stack, pre_partition, separator, post_partition);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::rsplit(str self, str separator=' ', int max=-1) -> str[]"),
        [](Stack& stack) {
          int64_t max = pop(stack).toInt();
          std::string separator = pop(stack).toStringRef();
          std::string string = pop(stack).toStringRef();
          std::reverse(separator.begin(), separator.end());
          std::reverse(string.begin(), string.end());

          std::string::size_type prev_pos = 0;
          std::string::size_type pos = 0;
          c10::List<std::string> splits;
          auto count = 0;
          while ((pos = string.find(separator, pos)) != std::string::npos) {
            count++;
            if (max >= 0 && count > max) {
              break;
            } else {
              auto substr = string.substr(prev_pos, pos - prev_pos);
              std::reverse(substr.begin(), substr.end());
              splits.emplace(splits.begin(), substr);
            }
            pos += separator.size();
            prev_pos = pos;
          }
          auto substr = string.substr(prev_pos, string.size() - prev_pos);
          std::reverse(substr.begin(), substr.end());
          splits.emplace(splits.begin(), substr);
          push(stack, std::move(splits));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::join(str self, str[] values) -> str"),
        [](Stack& stack) {
          IValue ivalue = pop(stack);
          c10::ArrayRef<IValue> ivalues = ivalue.toListRef();
          c10::List<std::string> values;
          for (const auto& v : ivalues) {
            values.emplace_back(v.toStringRef());
          }
          c10::optional<std::string> opt_string =
              pop(stack).toOptional<std::string>();
          const std::string& string = opt_string.value_or("");
          std::stringstream ss;
          for (auto it = values.begin(); it != values.end(); ++it) {
            ss << static_cast<std::string>(*it);
            if (it != values.end() - 1) {
              ss << string;
            }
          }
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),
};

RegisterOperators regStrOps(createOperators(stringOpGenArgs));

static const std::vector<OperatorGeneratorArgs> opGenArgs1{
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::rangelist(int n) -> int[]"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int64_t n;
          pop(stack, n);
          c10::List<int64_t> elems;
          elems.reserve(n);
          for (const auto i : c10::irange(n)) {
            elems.push_back(i);
          }
          push(stack, std::move(elems));
        },
        aliasAnalysisFromSchema()),
    // note: this op needs to share a name with the Scalar -> Tensor conversion
    // because all _to_tensor conversion have to have the same operator namet
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::NumToTensor.bool(bool a) -> Tensor"),
        numToTensorBool,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::device(str a) -> Device"),
        [](Stack& stack) {
          push(stack, c10::Device(pop(stack).toStringRef()));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::device.with_index(str type, int index) -> Device"),
        device_with_index,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::percentFormat(str self, ...) -> str"),
        [](Stack& stack) {
          size_t num_inputs = pop(stack).toInt();
          percentFormat(stack, num_inputs);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::to.prim_other(Tensor(a) self, bool non_blocking=False, bool copy=False) -> Tensor(a|b)"),
        [](Stack& stack) {
          at::Tensor self;
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          bool non_blocking;
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          bool copy;
          pop(stack, self, non_blocking, copy);
          c10::optional<c10::Device> device = c10::nullopt;
          c10::optional<at::ScalarType> scalarType = c10::nullopt;
          push(
              stack, to_dispatch(self, device, scalarType, non_blocking, copy));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::requires_grad(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.requires_grad());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::grad(Tensor a) -> Tensor(*)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.grad());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_sparse(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_sparse());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_sparse_csr(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_sparse_csr());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_mkldnn(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_mkldnn());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_mps(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_mps());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_vulkan(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_vulkan());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_ipu(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_ipu());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_quantized(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_quantized());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_meta(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_meta());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_ort(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_ort());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_nested(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_nested());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::name(Tensor a) -> str?"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          if (a.name().empty()) {
            push(stack, IValue());
          } else {
            push(stack, a.name());
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::nbytes(Tensor a) -> int"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          const auto nbytes = static_cast<int64_t>(a.nbytes());
          push(stack, nbytes);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::itemsize(Tensor a) -> int"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          const auto itemsize = static_cast<int64_t>(a.itemsize());
          push(stack, itemsize);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::index(Device self) -> int?"),
        [](Stack& stack) {
          auto d = pop(stack).toDevice();
          if (d.has_index()) {
            push(stack, d.index());
          } else {
            push(stack, IValue());
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        // TODO return generator object when torchscript supports RNG
        // first-class
        TORCH_SELECTIVE_SCHEMA("aten::manual_seed(int seed) -> ()"),
        [](Stack& stack) { at::manual_seed(pop(stack).toInt()); },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::cuda(Tensor(a) self) -> Tensor(a|b)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.cuda());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::AutogradZero() -> Tensor"),
        [](Stack& stack) { stack.emplace_back(at::Tensor()); },
        aliasAnalysisSpecialCase()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "prim::ReductionSizes(int[] size, int[] red_axes, bool keepdim = False) -> int[]"),
        [](Stack& stack) {
          bool keepdim = pop(stack).toBool();
          c10::List<int64_t> axes = pop(stack).toIntList();
          c10::List<int64_t> size = pop(stack).toIntList();
          if (keepdim) {
            for (const auto& axis : axes) {
              size.set(axis, 1);
            }
          } else {
            int64_t index = 0;
            auto iter = size.begin();
            std::sort(axes.begin(), axes.end());
            for (const auto& axis : axes) {
              // move iter to the next axis
              iter += axis - index;

              // input iter points to axis and is updated to axis + 1
              iter = size.erase(iter);

              // update current index for iter
              index = axis + 1;
            }
          }
          push(stack, IValue(std::move(size)));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::BroadcastSizes(...) -> int[]"),
        [](Stack& stack) {
          auto num_inputs = pop(stack).toInt();
          std::vector<int64_t> size;
          size.reserve(8);
          for (const auto i : c10::irange(num_inputs)) {
            size =
                at::infer_size(size, peek(stack, i, num_inputs).toDimVector());
          }
          drop(stack, num_inputs);
          push(stack, IValue(size));
        },
        aliasAnalysisSpecialCase()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::warn(str message, int stacklevel=2) -> ()"),
        [](Stack& stack) {
          TORCH_CHECK(false, "warn is implemented directly in the interpreter");
        },
        aliasAnalysisFromSchema()),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "onnx::Reshape(Tensor input, Tensor shape) -> Tensor"),
        [](Stack& stack) {
          at::Tensor input, shape;
          pop(stack, input, shape);
          shape = shape.contiguous();
          AT_ASSERT(shape.ndimension() == 1);
          at::IntArrayRef shape_list(
              shape.const_data_ptr<int64_t>(), shape.size(0));
          push(stack, input.reshape(shape_list));
        },
        aliasAnalysisSpecialCase()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("onnx::Shape(Tensor t) -> Tensor"),
        [](Stack& stack) {
          auto t = pop(stack).toTensor();
          at::IntArrayRef sizes = t.sizes();
          auto sizes_tensor = torch::empty(
              {static_cast<int64_t>(sizes.size())}, at::dtype(at::kLong));
          auto accessor = sizes_tensor.accessor<int64_t, 1>();
          for (const auto i : c10::irange(sizes.size())) {
            accessor[i] = sizes[i];
          }
          stack.emplace_back(sizes_tensor);
        },
        aliasAnalysisSpecialCase()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::AutogradAnyNonZero(...) -> bool"),
        [](Stack& stack) {
          auto num_inputs = pop(stack).toInt();
          bool result = false;
          for (const IValue& v : last(stack, num_inputs)) {
            if (v.isTensor()) {
              if (v.toTensor().defined()) {
                result = true;
                break;
              }
            } else if (v.isTensorList()) {
              for (const at::Tensor& t : v.toTensorVector()) {
                if (t.defined()) {
                  result = true;
                }
              }
              if (result) {
                break;
              }
            } else {
              TORCH_INTERNAL_ASSERT(false);
            }
          }
          drop(stack, num_inputs);
          stack.emplace_back(result);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::AutogradAllZero(...) -> bool"),
        [](Stack& stack) {
          auto num_inputs = pop(stack).toInt();
          bool result = true;
          for (const IValue& v : last(stack, num_inputs)) {
            TORCH_INTERNAL_ASSERT(v.isTensor());
            if (v.toTensor().defined()) {
              result = false;
              break;
            }
          }
          drop(stack, num_inputs);
          stack.emplace_back(result);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::AutogradAllNonZero(...) -> bool"),
        [](Stack& stack) {
          auto num_inputs = pop(stack).toInt();
          bool result = true;
          for (const IValue& v : last(stack, num_inputs)) {
            TORCH_INTERNAL_ASSERT(v.isTensor());
            if (!v.toTensor().defined()) {
              result = false;
              break;
            }
          }
          drop(stack, num_inputs);
          stack.emplace_back(result);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::AutogradAdd(Any a, Any b) -> Any"),
        [](Stack& stack) {
          IValue i_a = pop(stack);
          IValue i_b = pop(stack);
          if (i_a.isNone() && i_b.isNone()) {
            stack.emplace_back(at::Tensor{});
            return;
          }
          if (i_a.isNone()) {
            stack.emplace_back(i_b.toTensor());
            return;
          }
          if (i_b.isNone()) {
            stack.emplace_back(i_a.toTensor());
            return;
          }
          at::Tensor a = i_a.toTensor();
          at::Tensor b = i_b.toTensor();
          // NOLINTNEXTLINE(bugprone-branch-clone)
          if (!a.defined() && !b.defined()) {
            // undef + undef == undef
            stack.emplace_back(a);
          } else if (!a.defined()) {
            stack.emplace_back(b);
          } else if (!b.defined()) {
            stack.emplace_back(a);
          } else {
            stack.emplace_back(a + b);
          }
        },
        aliasAnalysisSpecialCase()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_size_if_not_equal(int[] self_size, int[] other_size) -> int[]?"),
        [](Stack& stack) {
          IValue self_size, other_size;
          pop(stack, self_size, other_size);
          auto s = self_size.toDimVector();
          auto o = other_size.toDimVector();
          if (s == o) {
            stack.emplace_back();
          } else {
            stack.emplace_back(std::move(self_size));
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_unwrap_optional(t(a)? optional) -> t(a)"),
        [](Stack& stack) {
          auto val = pop(stack);
          TORCH_CHECK(!val.isNone(), "Unwrapping null optional");
          push(stack, std::move(val));
        },
        aliasAnalysisFromSchema())};

RegisterOperators reg1(createOperators(opGenArgs1));

void hashValue(Stack& stack) {
  auto value = pop(stack);
  push(stack, value.hash());
}

static const std::vector<OperatorGeneratorArgs> opGenArgs2{
    // registered as Any[] so that heterogenous tuples can be called with len()
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::len.any(Any[] a) -> int"),
        listLen,
        aliasAnalysisFromSchema()),

// these ops have a specialized implementation for the list element type
#define CREATE_SPECIALIZED_LIST_OPS(decl_type, value_type) \
  OperatorGeneratorArgs(                                   \
      TORCH_SELECTIVE_SCHEMA(                              \
          "aten::remove." decl_type "(" decl_type          \
          "[](a!) self,                                                           \
        " decl_type " el) -> ()"),                         \
      listRemove<value_type>,                              \
      aliasAnalysisFromSchema()),                          \
      OperatorGeneratorArgs(                               \
          TORCH_SELECTIVE_SCHEMA(                          \
              "aten::index.list_" decl_type "(" decl_type  \
              "[] self,                                                               \
        " decl_type " el) -> int"),                        \
          listIndex<value_type>,                           \
          aliasAnalysisFromSchema()),                      \
      OperatorGeneratorArgs(                               \
          TORCH_SELECTIVE_SCHEMA(                          \
              "aten::count." decl_type "(" decl_type       \
              "[] self,                                                               \
        " decl_type " el) -> int"),                        \
          listCount<value_type>,                           \
          aliasAnalysisFromSchema()),

    CREATE_SPECIALIZED_LIST_OPS("int", int64_t)
        CREATE_SPECIALIZED_LIST_OPS("float", double)
            CREATE_SPECIALIZED_LIST_OPS("bool", bool)
                CREATE_SPECIALIZED_LIST_OPS("Tensor", at::Tensor)
                    CREATE_SPECIALIZED_LIST_OPS("str", std::string)

#undef CREATE_GENERIC_LIST_OPS
#undef CREATE_SPECIALIZED_LIST_OPS

    // `listContains<T>` is not implemented for non-primitive types
    // TODO: Add List[bool] once .to<c10::List<bool>> doesn't throw an error
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__contains__.float_list(float[] l, float item) -> bool"),
        listContains<double>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::sort.int(int[](a!) self, bool reverse=False) -> ()"),
        listSort<int64_t>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::sort.float(float[](a!) self, bool reverse=False) -> ()"),
        listSort<double>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::sort.Tensor(Tensor[](a!) self, bool reverse=False) -> ()"),
        listSort<at::Tensor>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::sort.bool(bool[](a!) self, bool reverse=False) -> ()"),
        listSort<bool>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::sort.str(str[](a!) self, bool reverse=False) -> ()"),
        listSort<std::string>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sorted.int(int[](a) input) -> (int[])"),
        listCopyAndSort<int64_t>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::sorted.float(float[](a) input) -> (float[])"),
        listCopyAndSort<double>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::sorted.Tensor(Tensor[](a) input) -> (Tensor[])"),
        listCopyAndSort<at::Tensor>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::sorted.bool(bool[](a) input) -> (bool[])"),
        listCopyAndSort<bool>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sorted.str(str[](a) input) -> (str[])"),
        listCopyAndSort<std::string>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::eq.float_list(float[] a, float[] b) -> bool"),
        listEq<double>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::eq.Tensor_list(Tensor[] a, Tensor[] b) -> bool"),
        listEq<at::Tensor>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::eq.bool_list(bool[] a, bool[] b) -> bool"),
        listEq<bool>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::eq.str_list(str[] a, str[] b) -> bool"),
        listEq<std::string>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::ne.float_list(float[] a, float[] b) -> bool"),
        listNe<double>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::ne.Tensor_list(Tensor[] a, Tensor[] b) -> bool"),
        listNe<at::Tensor>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::ne.bool_list(bool[] a, bool[] b) -> bool"),
        listNe<bool>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ne.str_list(str[] a, str[] b) -> bool"),
        listNe<std::string>,
        aliasAnalysisFromSchema()),

#define DEFINE_CONVERT_BASE_OP(op_name, prefix, char_op) \
  OperatorGeneratorArgs(                                 \
      TORCH_SELECTIVE_SCHEMA(#op_name "(int i) -> str"), \
      [](Stack& stack) {                                 \
        auto i = pop(stack).toInt();                     \
        std::stringstream ss;                            \
        if (i < 0) {                                     \
          ss << "-";                                     \
          i = -i;                                        \
        }                                                \
        ss << "0" << prefix << char_op << i;             \
        push(stack, ss.str());                           \
      },                                                 \
      aliasAnalysisFromSchema())

    DEFINE_CONVERT_BASE_OP(aten::hex, "x", std::hex),
    DEFINE_CONVERT_BASE_OP(aten::oct, "o", std::oct),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::bin(int i) -> str"),
        [](Stack& stack) {
          auto i = pop(stack).toInt();
          std::stringstream ss;
          if (i == 0) {
            push(stack, "0b0");
          } else {
            if (i < 0) {
              ss << "-";
              i = -i;
            }
            std::string str = std::bitset<8 * sizeof(i)>(i).to_string();
            str.erase(0, std::min(str.find_first_not_of('0'), str.size() - 1));
            ss << "0b" << str;
            push(stack, ss.str());
          }
        },
        aliasAnalysisFromSchema()),
    // TODO: deprecate this in favor of aten::getelem
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "prim::StringIndex(str string, int index) -> str"),
        [](Stack& stack) {
          auto index = pop(stack).toInt();
          auto string = pop(stack).toStringRef();
          auto norm_index = normalizeIndex(index, string.size());
          char c = string.at(norm_index);
          push(stack, std::string(&c, 1));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::chr(int i) -> str"),
        [](Stack& stack) {
          auto i = pop(stack).toInt();
          std::stringstream ss;
          TORCH_CHECK(
              i >= 0 && i < 1114111,
              "chr() arg not in range(0x110000), found ",
              i);
          char c = i;
          ss << c;
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),

    // only used in loop unrolling, not exposed to end users
    DEFINE_INT_OP(aten::__round_to_zero_floordiv, a / b),

    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::modf(float a) -> (float, float)"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          double a;
          pop(stack, a);
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          double b, c;
          b = modf(a, &c);
          push(stack, b, c);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::frexp(float a) -> (float, int)"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          double a;
          pop(stack, a);
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          double m;
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int e;
          m = std::frexp(a, &e);
          push(stack, m, e);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ldexp(float x, int i) -> float"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          double a;
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int64_t b;
          pop(stack, a, b);
          push(stack, std::ldexp(a, b));
        },
        aliasAnalysisFromSchema()),
    DEFINE_BINARY_FLOAT_OP(aten::mathremainder, std::remainder(a, b)),

    DEFINE_INT_OP(aten::__and__, a& b),
    DEFINE_INT_OP(aten::__or__, a | b),
    DEFINE_INT_OP(aten::__xor__, a ^ b),
    DEFINE_INT_OP(aten::__lshift__, a << b),
    DEFINE_INT_OP(aten::__rshift__, a >> b),

    DEFINE_GENERIC_BINARY_OP(
        aten::log,
        std::log(a) / std::log(b),
        float,
        complex),
    DEFINE_INT_FLOAT_OP(aten::log, std::log(a) / std::log(b), float),
    DEFINE_INT_COMPLEX_OP(aten::log, std::log(a) / std::log(b), complex),
    DEFINE_FLOAT_COMPLEX_OP(aten::log, std::log(a) / std::log(b), complex),
    DEFINE_SCALAR_BINARY_OP_AVOID_COLLISION(
        aten::log,
        std::log(a) / std::log(b),
        std::log(a) / std::log(b),
        float),
    DEFINE_UNARY_OP(aten::log1p, std::log1p(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::log10, std::log10(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::sqrt, std::sqrt(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::acos, std::acos(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::asin, std::asin(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::atan, std::atan(a), float, float),
    DEFINE_GENERIC_OP(
        aten::atan2,
        std::atan2(a, b),
        std::atan2(a, b),
        float,
        float),
    DEFINE_INT_FLOAT_OP(aten::atan2, std::atan2(a, b), float),
    DEFINE_SCALAR_BINARY_OP_AVOID_COLLISION(
        aten::atan2,
        std::atan2(a, b),
        std::atan2(a, b),
        float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::cos, std::cos(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::sin, std::sin(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::tan, std::tan(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::asinh, std::asinh(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::atanh, std::atanh(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::acosh, std::acosh(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::sinh, std::sinh(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::cosh, std::cosh(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::tanh, std::tanh(a), float, float),
    DEFINE_UNARY_OP_WITH_COMPLEX_CAST(
        aten::angle,
        std::arg(a),
        float,
        float,
        float,
        float),
    DEFINE_UNARY_OP(aten::degrees, degrees(a), float, float),
    DEFINE_UNARY_OP(aten::radians, radians(a), float, float),
    DEFINE_BINARY_FLOAT_OP(aten::fmod, std::fmod(a, b)),
    DEFINE_UNARY_INT_OP(aten::factorial, factorial(a), int),
    DEFINE_UNARY_FLOAT_OP(aten::isnan, std::isnan(a), bool),
    DEFINE_UNARY_FLOAT_OP(aten::isfinite, std::isfinite(a), bool),
    DEFINE_UNARY_FLOAT_OP(aten::isinf, std::isinf(a), bool),
    DEFINE_UNARY_COMPLEX_OP(
        aten::isnan,
        std::isnan(a.real()) || std::isnan(a.imag()),
        bool),
    DEFINE_UNARY_COMPLEX_OP(
        aten::isfinite,
        std::isfinite(a.real()) && std::isfinite(a.imag()),
        bool),
    DEFINE_UNARY_COMPLEX_OP(
        aten::isinf,
        std::isinf(a.real()) || std::isinf(a.imag()),
        bool),
    DEFINE_UNARY_OP(aten::gamma, std::tgamma(a), float, float),
    DEFINE_UNARY_OP(aten::erf, std::erf(a), float, float),
    DEFINE_UNARY_OP(aten::erfc, std::erfc(a), float, float),
    DEFINE_UNARY_OP(aten::expm1, std::expm1(a), float, float),
    DEFINE_UNARY_OP(aten::fabs, std::fabs(a), float, float),
    DEFINE_UNARY_OP(aten::lgamma, std::lgamma(a), float, float),

    // TODO: move abs to aten namespace because it's schematized!
    DEFINE_UNARY_OP_WITH_COMPLEX_CAST(
        prim::abs,
        std::abs(a),
        int,
        float,
        float,
        float),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::abs(Tensor x) -> Tensor"),
        [](Stack& stack) {
          at::Tensor x;
          pop(stack, x);
          push(stack, x.abs());
        },
        aliasAnalysisFromSchema()),

    DEFINE_INT_OP(aten::gcd, gcd(a, b)),

    DEFINE_GENERIC_OP(
        aten::copysign,
        std::copysign(a, b),
        std::copysign(a, b),
        float,
        float),
    DEFINE_INT_FLOAT_OP(aten::copysign, std::copysign(a, b), float),
    DEFINE_SCALAR_BINARY_OP(
        aten::copysign,
        std::copysign(a, b),
        std::copysign(a, b),
        float),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::_tensor_to_list(Tensor self) -> int[]"),
        [](Stack& stack) {
          at::Tensor t;
          pop(stack, t);
          c10::List<int64_t> elems;
          elems.reserve(t.size(0));
          for (const auto i : c10::irange(t.size(0))) {
            elems.push_back(*t[i].const_data_ptr<int32_t>());
          }
          push(stack, std::move(elems));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::_list_to_tensor(int[] self) -> Tensor"),
        [](Stack& stack) {
          c10::List<int64_t> l = pop(stack).toIntList();
          auto t = torch::empty(
              {static_cast<int64_t>(l.size())}, at::dtype(at::kInt));
          for (const auto i : c10::irange(l.size())) {
            t[i] = l.get(i);
          }
          push(stack, std::move(t));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sum.int(int[] self) -> int"),
        [](Stack& stack) {
          c10::List<int64_t> l = pop(stack).toIntList();
          auto sum = 0;
          for (const auto& elem : l) {
            sum += elem;
          }
          push(stack, sum);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sum.float(float[] self) -> float"),
        [](Stack& stack) {
          c10::List<double> l = pop(stack).toDoubleList();
          auto sum = 0.0;
          for (const auto& elem : l) {
            sum += elem;
          }
          push(stack, sum);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sum.complex(complex[] self) -> complex"),
        [](Stack& stack) {
          c10::List<c10::complex<double>> l = pop(stack).toComplexDoubleList();
          c10::complex<double> sum = 0.0;
          for (const auto i : c10::irange(l.size())) {
            sum = sum + l.extract(i);
          }
          push(stack, sum);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sum.bool(bool[] self) -> int"),
        [](Stack& stack) {
          c10::List<bool> l = pop(stack).toBoolList();
          auto sum = 0;
          for (const auto& elem : l) {
            if (elem) {
              sum += 1;
            }
          }
          push(stack, sum);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::any.str(str[] self) -> bool"),
        [](Stack& stack) {
          auto l = pop(stack).toList();
          for (const auto& elem : l) {
            if (elem != "") {
              push(stack, true);
              return;
            }
          }
          push(stack, false);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::any.int(int[] self) -> bool"),
        [](Stack& stack) {
          c10::List<int64_t> l = pop(stack).toIntList();
          for (const auto& elem : l) {
            if (elem) {
              push(stack, true);
              return;
            }
          }
          push(stack, false);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::any.float(float[] self) -> bool"),
        [](Stack& stack) {
          c10::List<double> l = pop(stack).toDoubleList();
          for (const auto& elem : l) {
            if (elem) {
              push(stack, true);
              return;
            }
          }
          push(stack, false);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::any.bool(bool[] self) -> bool"),
        [](Stack& stack) {
          c10::List<bool> l = pop(stack).toBoolList();
          for (const auto& elem : l) {
            if (elem) {
              push(stack, true);
              return;
            }
          }
          push(stack, false);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::all.int(int[] self) -> bool"),
        [](Stack& stack) {
          c10::List<int64_t> l = pop(stack).toIntList();
          for (const auto& elem : l) {
            if (!elem) {
              push(stack, false);
              return;
            }
          }
          push(stack, true);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::all.float(float[] self) -> bool"),
        [](Stack& stack) {
          c10::List<double> l = pop(stack).toDoubleList();
          for (const auto& elem : l) {
            if (!elem) {
              push(stack, false);
              return;
            }
          }
          push(stack, true);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::all.bool(bool[] self) -> bool"),
        [](Stack& stack) {
          c10::List<bool> l = pop(stack).toBoolList();
          for (const auto& elem : l) {
            if (!elem) {
              push(stack, false);
              return;
            }
          }
          push(stack, true);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::divmod.int(int x, int y) -> (int, int)"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int64_t a, b;
          lldiv_t divresult = {};
          pop(stack, a, b);
          if (b == 0) {
            throw std::runtime_error(
                "ZeroDivisionError: integer division or modulo by zero");
          }
          divresult = lldiv(a, b);
          if (divresult.rem && (a < 0) != (b < 0)) {
            divresult.quot -= 1;
            divresult.rem += b;
          }
          push(
              stack,
              static_cast<int64_t>(divresult.quot),
              static_cast<int64_t>(divresult.rem));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::divmod.float(float x, float y) -> (float, float)"),
        [](Stack& stack) {
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          double a, b;
          pop(stack, a, b);
          if (b == 0) {
            throw std::runtime_error("ZeroDivisionError: float divmod()");
          }
          double rem = fmod(a, b);
          // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
          if (rem && (a < 0) != (b < 0)) {
            rem += b;
          }
          push(stack, (a - rem) / b, rem);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::id(AnyClassType? x) -> int"),
        [](Stack& stack) {
          IValue a;
          pop(stack, a);
          if (a.isNone()) {
            push(stack, 0);
          } else {
            push(stack, reinterpret_cast<int64_t>(a.internalToPointer()));
          }
        },
        aliasAnalysisFromSchema()),
    // This operator is generated inside the compiler for indexing into
    // ModuleList without a statically determinable key. Accordingly,
    // self must be a ModuleType and the output must be an InterfaceType.
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "prim::ModuleContainerIndex.list(Any self, int ind) -> Any"),
        [](Stack& stack) {
          IValue ind = pop(stack);
          IValue module_dict = pop(stack);
          std::stringstream ss;
          ss << ind.toInt();
          push(
              stack, torch::jit::Object(module_dict.toObject()).attr(ss.str()));
        },
        aliasAnalysisFromSchema()),

#define DEFINE_DIVMOD_MIXED_OP(type_a, type_b)                               \
  OperatorGeneratorArgs(                                                     \
      TORCH_SELECTIVE_SCHEMA("aten::divmod." #type_a "_" #type_b "(" #type_a \
                             " x," #type_b " y) -> (float, float)"),         \
      [](Stack& stack) {                                                     \
        type_a a;                                                            \
        type_b b;                                                            \
        pop(stack, a, b);                                                    \
        if (b == 0) {                                                        \
          throw std::runtime_error("ZeroDivisionError: float divmod()");     \
        }                                                                    \
        double quot = floor(a / b);                                          \
        double rem = a - (quot * b);                                         \
        push(stack, quot, rem);                                              \
      },                                                                     \
      aliasAnalysisFromSchema())

    DEFINE_DIVMOD_MIXED_OP(int, float),
    DEFINE_DIVMOD_MIXED_OP(float, int),

#undef DEFINE_DIVMOD_MIXED_OP
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::hash.generic(t value) -> int"),
        hashValue,
        aliasAnalysisFromSchema()),

#define DEFINE_COMPLEX_OP(type_a, type_b, actual_type_a, actual_type_b)       \
  OperatorGeneratorArgs(                                                      \
      TORCH_SELECTIVE_SCHEMA("aten::Complex." #type_a "_" #type_b "(" #type_a \
                             " x," #type_b " y) -> complex"),                 \
      [](Stack& stack) {                                                      \
        actual_type_a a;                                                      \
        actual_type_b b;                                                      \
        pop(stack, a, b);                                                     \
        auto comp = c10::complex<double>(a, b);                               \
        push(stack, comp);                                                    \
      },                                                                      \
      aliasAnalysisFromSchema())

#define DEFINE_COMPLEX_OP_WITH_TENSOR_ARG(                                    \
    type_a, type_b, actual_type_a, actual_type_b)                             \
  OperatorGeneratorArgs(                                                      \
      TORCH_SELECTIVE_SCHEMA("aten::Complex." #type_a "_" #type_b "(" #type_a \
                             " x," #type_b " y) -> complex"),                 \
      [](Stack& stack) {                                                      \
        actual_type_a a;                                                      \
        actual_type_b b;                                                      \
        pop(stack, a, b);                                                     \
        auto comp = c10::complex<double>(a.item<double>(), b);                \
        push(stack, comp);                                                    \
      },                                                                      \
      aliasAnalysisFromSchema()),                                             \
      OperatorGeneratorArgs(                                                  \
          TORCH_SELECTIVE_SCHEMA("aten::Complex." #type_b "_" #type_a         \
                                 "(" #type_b " x," #type_a " y) -> complex"), \
          [](Stack& stack) {                                                  \
            actual_type_b a;                                                  \
            actual_type_a b;                                                  \
            pop(stack, a, b);                                                 \
            auto comp = c10::complex<double>(a, b.item<double>());            \
            push(stack, comp);                                                \
          },                                                                  \
          aliasAnalysisFromSchema())

    DEFINE_COMPLEX_OP(int, bool, int, bool),
    DEFINE_COMPLEX_OP(bool, int, bool, int),
    DEFINE_COMPLEX_OP(float, bool, double, bool),
    DEFINE_COMPLEX_OP(bool, float, bool, double),
    DEFINE_COMPLEX_OP(float, int, double, int),
    DEFINE_COMPLEX_OP(int, float, int, double),
    DEFINE_COMPLEX_OP(int, int, int, int),
    DEFINE_COMPLEX_OP(bool, bool, bool, bool),
    DEFINE_COMPLEX_OP(float, float, double, double),
    DEFINE_COMPLEX_OP_WITH_TENSOR_ARG(Tensor, float, at::Tensor, double),
    DEFINE_COMPLEX_OP_WITH_TENSOR_ARG(Tensor, int, at::Tensor, int),
    DEFINE_COMPLEX_OP_WITH_TENSOR_ARG(Tensor, bool, at::Tensor, bool),
};

RegisterOperators reg2(createOperators(opGenArgs2));

} // namespace
} // namespace jit
} // namespace torch
