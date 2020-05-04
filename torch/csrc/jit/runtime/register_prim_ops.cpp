#include <torch/csrc/jit/runtime/register_ops_utils.h>

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
    {Operator(
         // note the compiler knows to type TupleIndex more accurately than it
         // is listed here.
         "prim::TupleIndex(Any tup, int i) -> Any",
         [](Stack& stack) {
           int64_t index = pop(stack).toInt();
           auto tuple = pop(stack).toTuple();
           auto norm_index = normalizeIndex(index, tuple->elements().size());
           if (norm_index < 0 ||
               norm_index > static_cast<int64_t>(tuple->elements().size())) {
             throw std::out_of_range("Tuple list index out of range");
           }
           stack.emplace_back(tuple->elements()[norm_index]);
           return 0;
         },
         aliasAnalysisSpecialCase()),
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
             std::stringstream error_str;
             error_str << "could not convert string "
                       << "to float: '" << s->string() << "'";
             throw std::runtime_error(error_str.str());
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
         "prim::RaiseException(str msg) -> ()",
         [](Stack& stack) {
           throw JITException(pop(stack).toStringRef());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::Size(int[] sizes) -> int[]",
         [](Stack& stack) { return 0; },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::size(Tensor self) -> int[]",
         [](Stack& stack) {
           auto t = std::move(pop(stack)).toTensor();
           pack(stack, t.sizes().vec());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         // note the compiler knows to type TupleIndex more accurately than it
         // is listed here.
         "prim::TupleIndex(Any tup, int i) -> Any",
         [](Stack& stack) {
           int64_t index = pop(stack).toInt();
           auto tuple = pop(stack).toTuple();
           auto norm_index = normalizeIndex(index, tuple->elements().size());
           if (norm_index < 0 ||
               norm_index > static_cast<int64_t>(tuple->elements().size())) {
             throw std::out_of_range("Tuple list index out of range");
           }
           stack.emplace_back(tuple->elements()[norm_index]);
           return 0;
         },
         aliasAnalysisSpecialCase()),
     Operator(
         "aten::ne.int_list(int[] a, int[] b) -> bool",
         listNe<int64_t>,
         aliasAnalysisFromSchema()),
     Operator(
         "prim::unchecked_unwrap_optional(t(a)? optional) -> t(a)",
         noop,
         aliasAnalysisFromSchema()),
     Operator(
         "prim::device(Tensor a) -> Device",
         [](Stack& stack) {
           push(stack, pop(stack).toTensor().device());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::dtype(Tensor a) -> int",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, static_cast<int64_t>(a.scalar_type()));
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
     Operator(
         "aten::element_size(Tensor self) -> int",
         [](Stack& stack) {
           at::Tensor arg = pop(stack).toTensor();
           push(stack, arg.element_size());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::numel(Tensor self) -> int",
         [](Stack& stack) {
           at::Tensor arg = pop(stack).toTensor();
           push(stack, arg.numel());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::dim(Tensor self) -> int",
         [](Stack& stack) {
           at::Tensor arg = pop(stack).toTensor();
           push(stack, arg.dim());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::get_device(Tensor self) -> int",
         [](Stack& stack) {
           RECORD_FUNCTION("get_device", std::vector<c10::IValue>());
           auto result =
               at::get_device((std::move(peek(stack, 0, 1))).toTensor());
           drop(stack, 1);
           pack(stack, result);
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::storage_offset(Tensor self) -> int",
         [](Stack& stack) {
           RECORD_FUNCTION("storage_offset", std::vector<c10::IValue>());
           auto result =
               ((std::move(peek(stack, 0, 1))).toTensor()).storage_offset();
           drop(stack, 1);
           pack(stack, result);
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::is_contiguous(Tensor self) -> bool",
         [](Stack& stack) {
           RECORD_FUNCTION("is_contiguous", std::vector<c10::IValue>());
           auto result =
               ((std::move(peek(stack, 0, 1))).toTensor()).is_contiguous();
           drop(stack, 1);
           pack(stack, result);
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
     Operator(
         "aten::list.t(t[] l) -> t[]",
         listList,
         aliasAnalysisFromSchema()),
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
     Operator(
         "prim::Uninitialized() -> Any",
         [](Stack& stack) {
           push(stack, IValue::uninitialized());
           return 0;
         },
         aliasAnalysisSpecialCase()),
     Operator(
         "prim::Print(...) -> ()",
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
           return 0;
         },
         aliasAnalysisSpecialCase()),
     DEFINE_COMPARISON_OP(aten::eq, a == b),
     DEFINE_COMPARISON_OP(aten::ne, a != b),
     DEFINE_COMPARISON_OP(aten::lt, a < b),
     DEFINE_COMPARISON_OP(aten::gt, a > b),
     DEFINE_COMPARISON_OP(aten::le, a <= b),
     DEFINE_COMPARISON_OP(aten::ge, a >= b),
     DEFINE_BINARY_OP(aten::add, a + b),
     DEFINE_BINARY_OP(aten::sub, a - b),
     DEFINE_BINARY_OP(aten::mul, a* b),

     //
     // create a clone of these declarations with a _hacked_twin overload name
     // and nullability scrubbed from TensorList arg types
     // TOOD find out why this exists and how to do it without the hack
     //
     Operator(
         "aten::index.Tensor_hacked_twin(Tensor self, Tensor[] indices) -> Tensor",
         [](Stack& stack) {
           auto indices = pop(stack).toTensorVector();
           auto self = pop(stack).toTensor();
           auto result = at::index(self, indices);
           push(stack, std::move(result));
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::_index_put_impl_.hacked_twin(Tensor(a!) self, Tensor[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)",
         [](Stack& stack) {
           auto unsafe = pop(stack).toBool();
           auto accumulate = pop(stack).toBool();
           auto values = pop(stack).toTensor();
           auto indices = pop(stack).toTensorVector();
           auto self = pop(stack).toTensor();
           auto result =
               at::_index_put_impl_(self, indices, values, accumulate, unsafe);
           push(stack, std::move(result));
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::index_put_.hacked_twin(Tensor(a!) self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)",
         [](Stack& stack) {
           auto accumulate = pop(stack).toBool();
           auto values = pop(stack).toTensor();
           auto indices = pop(stack).toTensorVector();
           auto self = pop(stack).toTensor();
           auto result = at::index_put_(self, indices, values, accumulate);
           push(stack, std::move(result));
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::index_put.hacked_twin(Tensor self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor",
         [](Stack& stack) {
           auto accumulate = pop(stack).toBool();
           auto values = pop(stack).toTensor();
           auto indices = pop(stack).toTensorVector();
           auto self = pop(stack).toTensor();
           auto result = at::index_put_(self, indices, values, accumulate);
           push(stack, std::move(result));
           return 0;
         },
         aliasAnalysisFromSchema())});

int dictSetItem(Stack& stack) {
  auto value = pop(stack);
  auto idx = pop(stack);
  auto dict = pop(stack).toGenericDict();
  dict.insert_or_assign(std::move(idx), std::move(value));
  return 0;
}

int dictLen(Stack& stack) {
  auto dict = pop(stack).toGenericDict();
  push(stack, int64_t(dict.size()));
  return 0;
}

int dictValues(Stack& stack) {
  auto dict = pop(stack).toGenericDict();
  auto values = c10::impl::GenericList(dict.valueType());
  for (const auto& entry : dict) {
    values.emplace_back(entry.value());
  }
  push(stack, values);
  return 0;
}

int dictKeys(Stack& stack) {
  auto dict = pop(stack).toGenericDict();
  auto keys = c10::impl::GenericList(dict.keyType());
  for (const auto& entry : dict) {
    keys.emplace_back(entry.key());
  }
  push(stack, keys);
  return 0;
}

int dictIndex(Stack& stack) {
  auto key = pop(stack);
  auto dict = pop(stack).toGenericDict();
  auto value = dict.find(key);
  if (value == dict.end()) {
    AT_ERROR("KeyError: ", key);
  }
  push(stack, value->value());
  return 0;
}

template <bool has_default>
int dictGet(Stack& stack) {
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
  return 0;
}

// If the key is in the dict, return it. Else set it to the default value and
// return that.
int dictSetDefault(Stack& stack) {
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
  return 0;
}

template <bool has_default>
int dictPop(Stack& stack) {
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
  return 0;
}

int dictDelete(Stack& stack) {
  dictPop<false>(stack);
  // pop pushes an item on the stack but delete does not, so get rid of it
  pop(stack);
  return 0;
}

int dictPopItem(Stack& stack) {
  auto dict = pop(stack).toGenericDict();
  if (dict.size() == 0) {
    AT_ERROR("popitem(): dictionary is empty");
  }
  auto head_item = dict.begin();

  IValue tuple =
      c10::ivalue::Tuple::create({head_item->key(), head_item->value()});
  auto erase_count = dict.erase(head_item->key());
  TORCH_CHECK(
      erase_count == 1, "Expected to erase 1 item, found ", erase_count);
  push(stack, tuple);
  return 0;
}

int dictContains(Stack& stack) {
  auto key = pop(stack);
  auto dict = pop(stack).toGenericDict();
  push(stack, dict.contains(key));
  return 0;
}

int dictClear(Stack& stack) {
  auto dict = pop(stack).toGenericDict();
  dict.clear();
  return 0;
}

int dictUpdate(Stack& stack) {
  auto to_add = pop(stack).toGenericDict();
  auto dict = pop(stack).toGenericDict();

  for (const auto& item : to_add) {
    dict.insert(item.key(), item.value());
  }
  return 0;
}

int dictItems(Stack& stack) {
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
  return 0;
}

int dictCopy(Stack& stack) {
  push(stack, pop(stack).toGenericDict().copy());
  return 0;
}

int dictConstructFromList(Stack& stack) {
  auto input_list = pop(stack);
  auto list = input_list.toList();
  auto tup_type = list.elementType()->expect<TupleType>();
  auto dict = c10::impl::GenericDict(
      tup_type->elements().at(0), tup_type->elements().at(1));
  dict.reserve(list.size());
  for (IValue input : list) {
    const auto tup = input.toTuple()->elements();
    dict.insert_or_assign(tup[0], tup[1]);
  }
  push(stack, dict);
  return 0;
}

#define CREATE_DICT_OPS(key_type)                                            \
  Operator(                                                                  \
      "aten::len.Dict_" key_type "(Dict(" key_type ", t) self) -> int",      \
      dictLen,                                                               \
      aliasAnalysisFromSchema()),                                            \
      Operator(                                                              \
          "aten::keys." key_type "(Dict(" key_type ", t) self) -> " key_type \
          "[](*)",                                                           \
          dictKeys,                                                          \
          aliasAnalysisFromSchema()),                                        \
      Operator(                                                              \
          "aten::values." key_type "(Dict(" key_type ", t) self) -> t[](*)", \
          dictValues,                                                        \
          aliasAnalysisFromSchema()),                                        \
      Operator(                                                              \
          "aten::__getitem__.Dict_" key_type "(Dict(" key_type               \
          ", t) self, " key_type " key) -> t(*)",                            \
          dictIndex,                                                         \
          aliasAnalysisFromSchema()),                                        \
      Operator(                                                              \
          "aten::get." key_type "(Dict(" key_type ", t) self, " key_type     \
          " key) -> t(*)?",                                                  \
          dictGet<false>,                                                    \
          aliasAnalysisFromSchema()),                                        \
      Operator(                                                              \
          "aten::get.default_" key_type "(Dict(" key_type                    \
          ", t) self, " key_type " key, t default_value) -> t(*)",           \
          dictGet<true>,                                                     \
          aliasAnalysisFromSchema()),                                        \
      Operator(                                                              \
          "aten::setdefault." key_type "(Dict(" key_type                     \
          ", t)(a!) self, " key_type                                         \
          "(b -> *) key, t(c -> *) default_value) -> t(*)",                  \
          dictSetDefault,                                                    \
          aliasAnalysisFromSchema()),                                        \
      Operator(                                                              \
          "aten::Delete.Dict_" key_type "(Dict(" key_type                    \
          ", t)(a!) self, " key_type " key) -> ()",                          \
          dictDelete,                                                        \
          aliasAnalysisFromSchema()),                                        \
      Operator(                                                              \
          "aten::pop.Dict_" key_type "(Dict(" key_type                       \
          ", t)(a!) self, " key_type " key) -> t(*)",                        \
          dictPop<false>,                                                    \
          aliasAnalysisFromSchema()),                                        \
      Operator(                                                              \
          "aten::pop.Dict_default_" key_type "(Dict(" key_type               \
          ", t)(a!) self, " key_type " key, t default_value) -> t(*)",       \
          dictPop<true>,                                                     \
          aliasAnalysisFromSchema()),                                        \
      Operator(                                                              \
          "aten::popitem." key_type "(Dict(" key_type                        \
          ", t)(a!) self) -> ((" key_type ", t))",                           \
          dictPopItem,                                                       \
          aliasAnalysisFromSchema()),                                        \
      Operator(                                                              \
          "aten::clear." key_type "(Dict(" key_type ", t)(a!) self) -> ()",  \
          dictClear,                                                         \
          aliasAnalysisFromSchema()),                                        \
      Operator(                                                              \
          "aten::update." key_type "(Dict(" key_type                         \
          ", t)(a!) self, Dict(" key_type ", t)(a!) to_add) -> ()",          \
          dictUpdate,                                                        \
          aliasAnalysisFromSchema()),                                        \
      Operator(                                                              \
          "aten::items." key_type "(Dict(" key_type                          \
          ", t) self) -> ((" key_type ", t)[])",                             \
          dictItems,                                                         \
          aliasAnalysisFromSchema()),                                        \
      Operator(                                                              \
          "aten::copy.Dict_" key_type "(Dict(" key_type                      \
          ", t)(a) self) -> Dict(" key_type ", t)",                          \
          dictCopy,                                                          \
          aliasAnalysisFromSchema()),                                        \
      Operator(                                                              \
          "aten::__contains__." key_type "(Dict(" key_type                   \
          ", t) dict, " key_type " key) -> bool",                            \
          dictContains,                                                      \
          aliasAnalysisFromSchema()),                                        \
      Operator(                                                              \
          "aten::_set_item." key_type "(Dict(" key_type                      \
          ", t)(a!) l, " key_type "(b -> *) idx, t(c -> *) v) -> ()",        \
          dictSetItem,                                                       \
          aliasAnalysisFromSchema()),                                        \
      Operator(                                                              \
          "aten::dict." key_type "((" key_type                               \
          ", tVal)[] inputs) -> Dict(" key_type ", tVal)",                   \
          dictConstructFromList,                                             \
          aliasAnalysisFromSchema())

RegisterOperators reg_dict_ops({
    CREATE_DICT_OPS("str"),
    CREATE_DICT_OPS("int"),
    CREATE_DICT_OPS("float"),
    CREATE_DICT_OPS("Tensor"),
});

} // namespace
} // namespace jit
} // namespace torch
