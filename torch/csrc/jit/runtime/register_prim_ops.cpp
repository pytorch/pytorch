#include "register_ops_utils.h"

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

RegisterOperators reg({
    Operator(
        "prim::TupleUnpack(Any tup) -> ...",
        [](Stack& stack) {
          tupleUnpack(stack);
          return 0;
        },
        aliasAnalysisSpecialCase()),
    Operator(
        "prim::unchecked_cast(t x) -> t",
        noop,
        aliasAnalysisSpecialCase()),
    Operator(
        "aten::IntImplicit(Tensor a) -> int",
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          checkImplicitTensorToNum(a, /*to int*/ true);
          push(stack, a.item<int64_t>());
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::FloatImplicit(Tensor a) -> float",
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          checkImplicitTensorToNum(a, /*to int*/ false);
          push(stack, a.item<double>());
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::ScalarImplicit(Tensor a) -> Scalar",
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          checkImplicitTensorToNum(a, /*to int*/ false);
          push(stack, a.item());
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::Bool.Tensor(Tensor a) -> bool",
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_nonzero());
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::Bool.int(int a) -> bool",
        [](Stack& stack) {
          int64_t i;
          pop(stack, i);
          push(stack, (bool)i);
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::Bool.float(float a) -> bool",
        [](Stack& stack) {
          double d;
          pop(stack, d);
          push(stack, (bool)d);
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::Float.Tensor(Tensor a) -> float",
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.item<double>());
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::Float.Scalar(Scalar a) -> float",
        [](Stack& stack) {
          IValue scalar;
          pop(stack, scalar);
          if (scalar.isDouble()) {
            push(stack, std::move(scalar));
          } else {
            push(stack, static_cast<double>(scalar.toInt()));
          }
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::Float.int(int a) -> float",
        [](Stack& stack) {
          int64_t i;
          pop(stack, i);
          push(stack, (float)i);
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::Float.bool(bool a) -> float",
        [](Stack& stack) {
          bool b;
          pop(stack, b);
          push(stack, (float)b);
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::Float.str(str a) -> float",
        [](Stack& stack) {
          auto s = pop(stack).toString();
          std::string::size_type sz;
          double b = c10::stod(s->string(), &sz);
          if (sz == s->string().size()) {
            push(stack, b);
          } else {
            throw std::runtime_error(
                "float() only accepts a string of single float number");
          }
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::format(str self, ...) -> str",
        [](Stack& stack) {
          size_t num_inputs = pop(stack).toInt();
          format(stack, num_inputs);
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "prim::NumToTensor.Scalar(Scalar a) -> Tensor",
        [](Stack& stack) {
          at::Scalar s;
          pop(stack, s);
          push(stack, at::scalar_to_tensor(s));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__not__(bool self) -> bool",
        [](Stack& stack) {
          push(stack, !pop(stack).toBool());
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__is__(t1 self, t2 obj) -> bool",
        [](Stack& stack) {
          IValue self, obj;
          pop(stack, self, obj);
          push(stack, self.is(obj));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__isnot__(t1 self, t2 obj) -> bool",
        [](Stack& stack) {
          IValue self, obj;
          pop(stack, self, obj);
          push(stack, !self.is(obj));
          return 0;
        },
        aliasAnalysisFromSchema()),
    // these ops are generic over the list element type.
    // CREATING GENERIC_LIST_OPS
    Operator(
        "aten::select.t(t[](a) list, int idx) -> t(*)",
        listSelect,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__getitem__.t(t[](a) list, int idx) -> t(*)",
        listSelect,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::append.t(t[](a!) self, t(c -> *) el) -> t[](a!)",
        listAppend,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::reverse.t(t[](a!) self) -> ()",
        listReverse,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::extend.t(t[](a!) self, t[] other) -> ()",
        listExtend,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::copy.t(t[](a) self) -> t[]",
        listCopy,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::_set_item.t(t [](a!) l, int idx, t(b -> *) el) -> t[](a!)",
        listSetItem,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::clear.t(t[](a!) self) -> ()",
        listClear,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::Delete.t(t[](a!) self, int idx) -> ()",
        listDelete,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::insert.t(t[](a!) self, int idx, t(b -> *) el) -> ()",
        listInsert,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::pop.t(t[](a!) self, int idx=-1) -> t(*)",
        listPop,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::add.t(t[] a, t[] b) -> t[]",
        listAdd,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::add_.t(t[](a!) self, t[] b) -> t[]",
        listInplaceAdd,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::slice.t(t[] l, int start, int end=9223372036854775807, int step=1) -> t[]",
        listSlice,
        aliasAnalysisFromSchema()),
    Operator("aten::list.t(t[] l) -> t[]", listList, aliasAnalysisFromSchema()),
    Operator(
        "aten::mul.left_t(t[] l, int n) -> t[]",
        listMulIntLeft,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::mul.right_(int n, t[] l) -> t[]",
        listMulIntRight,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::mul_.t(t[](a!) l, int n) -> t[](a!)",
        listMulIntLeftInPlace,
        aliasAnalysisFromSchema()),
    Operator("aten::len.t(t[] a) -> int", listLen, aliasAnalysisFromSchema()),
    DEFINE_COMPARISON_OP(aten::eq, a == b),
    DEFINE_COMPARISON_OP(aten::ne, a != b),
    DEFINE_COMPARISON_OP(aten::lt, a < b),
    DEFINE_COMPARISON_OP(aten::gt, a > b),
    DEFINE_COMPARISON_OP(aten::le, a <= b),
    DEFINE_COMPARISON_OP(aten::ge, a >= b),
});

} // namespace
} // namespace jit
} // namespace torch
