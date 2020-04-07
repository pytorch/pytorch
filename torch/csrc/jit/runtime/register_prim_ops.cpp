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


RegisterOperators reg(
    {
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
