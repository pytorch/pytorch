#include <torch/csrc/jit/ir/constants.h>

#include <ATen/core/functional.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>

namespace torch::jit {

static bool insertableTensor(const at::Tensor& ten) {
  // bail if tensor has no storage i.e. opaque tensor used in MKLdnn.
  // or gradients because we have no way of serializing them & are mutable
  return !ten.requires_grad() && ten.has_storage() && !ten.is_nested();
}

static bool insertableIValue(const IValue& ivalue) {
  if (ivalue.isInt() || ivalue.isNone() || ivalue.isBool() ||
      ivalue.isDouble() || ivalue.isComplexDouble() || ivalue.isString() ||
      ivalue.isDevice() || ivalue.isEnum()) {
    return true;
  }
  if (ivalue.isTensor()) {
    return insertableTensor(ivalue.toTensor());
  }
  if (ivalue.isList() || ivalue.isTuple()) {
    c10::ArrayRef<IValue> elems;
    if (ivalue.isTuple()) {
      elems = ivalue.toTupleRef().elements();
    } else {
      elems = ivalue.toListRef();
    }
    return std::all_of(elems.begin(), elems.end(), [](const IValue& tup_elem) {
      return insertableIValue(tup_elem);
    });
  }
  if (ivalue.isGenericDict()) {
    const auto& dict = ivalue.toGenericDict();
    return std::all_of(dict.begin(), dict.end(), [](const auto& entry) {
      return insertableIValue(entry.key()) && insertableIValue(entry.value());
    });
  }

  return false;
}

Value* insertConstant(
    Graph& g,
    const IValue& val,
    c10::optional<SourceRange> loc,
    c10::optional<ScopePtr> scope) {
  auto value = tryInsertConstant(g, val, std::move(loc), std::move(scope));
  if (value) {
    return *value;
  }
  throw constant_not_supported_error(
      "Unsupported value kind: " + val.tagKind());
}

// IValue -> Constant node
c10::optional<Value*> tryInsertConstant(
    Graph& g,
    const IValue& val,
    c10::optional<SourceRange> loc,
    c10::optional<ScopePtr> scope) {
  Node* n = g.create(prim::Constant);
  if (val.isTensor()) {
    at::Tensor ref = val.toTensor();
    if (!insertableTensor(val.toTensor())) {
      n->destroy();
      return c10::nullopt;
    }
    if (!ref.defined()) {
      n->destroy();
      return g.insertNode(g.createNone())->output();
    }
    TORCH_INTERNAL_ASSERT(!ref.requires_grad());
    n->output()->inferTypeFrom(
        ref); // note: before t_ because of std::move(ref)
    n->t_(attr::value, std::move(ref));
  } else if (val.isInt()) {
    n->i_(attr::value, val.toInt());
    n->output()->setType(IntType::get());
  } else if (val.isDouble()) {
    n->f_(attr::value, val.toDouble());
    n->output()->setType(FloatType::get());
  } else if (val.isComplexDouble()) {
    n->c_(attr::value, val.toComplexDouble());
    n->output()->setType(ComplexType::get());
  } else if (val.isBool()) {
    n->i_(attr::value, val.toBool());
    n->output()->setType(BoolType::get());
  } else if (val.isList()) {
    bool fast_path_list =
        val.isBoolList() || val.isIntList() || val.isDoubleList();
    if (fast_path_list || insertableIValue(val)) {
      n->ival_(attr::value, val);
      n->output()->setType(val.type());
    } else {
      n->destroy();
      return c10::nullopt;
    }
  } else if (val.isString()) {
    n->s_(attr::value, val.toStringRef());
    n->output()->setType(StringType::get());
  } else if (val.isDevice()) {
    std::stringstream ss;
    ss << val.toDevice();
    n->s_(attr::value, ss.str());
    n->output()->setType(DeviceObjType::get());
  } else if (val.isGenerator()) {
    auto generator = val.toGenerator();
    n->ival_(attr::value, generator);
    n->output()->setType(GeneratorType::get());
  } else if (val.isStream()) {
    // packing into int64_t removed
    n->ival_(attr::value, val);
    n->output()->setType(StreamObjType::get());
  } else if (val.isNone()) {
    n->output()->setType(NoneType::get());
  } else if (val.isTuple()) {
    if (insertableIValue(val)) {
      n->ival_(attr::value, val);
      n->output()->setType(val.type());
    } else {
      n->destroy();
      return c10::nullopt;
    };
  } else if (val.isObject()) {
    const auto& ref = val.toObjectRef();
    // see: [Constant Object Weak CompilationUnit Reference]
    if (!ref.type()->is_module() &&
        (ref.is_weak_compilation_ref() ||
         ref.is_empty_strong_compilation_ref())) {
      n->ival_(attr::value, val);
      n->output()->setType(val.type());
    } else {
      n->destroy();
      return c10::nullopt;
    }
  } else if ((val.isGenericDict() && insertableIValue(val)) || (val.isEnum())) {
    n->ival_(attr::value, val);
    n->output()->setType(val.type());
  } else {
    n->destroy();
    return c10::nullopt;
  }
  if (loc)
    n->setSourceRange(*loc);
  if (scope)
    n->setScope(*scope);
  return g.insertNode(n)->output();
}

c10::optional<IValue> toIValue(const Value* v) {
  if (v->node()->kind() != prim::Constant || v->type()->cast<FunctionType>()) {
    return c10::nullopt;
  }
  const Node* node = v->node();
  const TypePtr& type = v->type();
  if (type->isSubtypeOf(*TensorType::get())) {
    return node->t(attr::value);
  } else if (type->isSubtypeOf(*BoolType::get())) {
    return (bool)node->i(attr::value);
  } else if (
      type->isSubtypeOf(*NumberType::get()) &&
      node->kindOf(attr::value) == AttributeKind::i) {
    return node->i(attr::value);
  } else if (
      type->isSubtypeOf(*NumberType::get()) &&
      node->kindOf(attr::value) == AttributeKind::f) {
    return node->f(attr::value);
  } else if (
      type->isSubtypeOf(*NumberType::get()) &&
      node->kindOf(attr::value) == AttributeKind::c) {
    return node->c(attr::value);
  } else if (
      type->cast<ListType>() &&
      node->kindOf(attr::value) == AttributeKind::ival) {
    const auto& list = node->ival(attr::value);
    TORCH_INTERNAL_ASSERT(list.isList());
    return list;
  } else if (
      type->cast<DictType>() &&
      node->kindOf(attr::value) == AttributeKind::ival) {
    const auto& dict = node->ival(attr::value);
    TORCH_INTERNAL_ASSERT(dict.isGenericDict());
    return dict;
  } else if (
      type->cast<TupleType>() &&
      node->kindOf(attr::value) == AttributeKind::ival) {
    const auto& tup = node->ival(attr::value);
    TORCH_INTERNAL_ASSERT(tup.isTuple());
    return tup;
  } else if (type == StringType::get()) {
    const auto& s = node->s(attr::value);
    return s;
  } else if (type == DeviceObjType::get()) {
    auto d = c10::Device(node->s(attr::value));
    return d;
  } else if (type == GeneratorType::get()) {
    auto generator = node->ival(attr::value).toGenerator();
    return generator;
  } else if (type == StreamObjType::get()) {
    // int64_t packing removed
    auto s = node->ival(attr::value).toStream();
    return s;
  } else if (node->mustBeNone()) {
    return IValue();
  } else if (type->cast<EnumType>()) {
    const auto& enum_val = node->ival(attr::value);
    return enum_val;
  } else if (type->cast<ClassType>() && !type->is_module()) {
    const auto& class_val = node->ival(attr::value);
    return class_val;
  } else {
    std::stringstream ss;
    ss << "constant literal not supported for: " << type->str();
    throw std::runtime_error(ss.str());
  }
}

} // namespace torch::jit
