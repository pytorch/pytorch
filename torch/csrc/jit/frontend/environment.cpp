#include <torch/csrc/jit/frontend/environment.h>

namespace torch {
namespace jit {

void Environment::setVariableTypeError(
    const std::string& name,
    std::function<std::string()> msg) {
  auto runner = this;
  while (runner->next) {
    runner = runner->next.get();
  }
  runner->error_messages[name] = std::move(msg);
}

c10::optional<std::string> Environment::findVariableTypeError(
    const std::string& name) {
  auto runner = this;
  while (runner->next) {
    runner = runner->next.get();
  }
  auto msg = runner->error_messages.find(name);
  if (msg != runner->error_messages.end()) {
    return msg->second();
  } else {
    return c10::nullopt;
  }
}

SugaredValuePtr Environment::insertLoad(
    const std::string& name,
    const TypePtr& type) {
  auto g = b->owningGraph();
  auto load = g->insertNode(g->createLoad(name, type));
  if (meaningfulName(name)) {
    load->output()->setDebugName(name);
  }
  return std::make_shared<SimpleValue>(load->output());
}

// note: type is not always the same as v->type(), e.g.
// type: Optional[Tensor]
// v->type(): Tensor
void Environment::insertStore(
    const std::string& name,
    const SourceRange& loc,
    Value* v,
    TypePtr type) {
  auto g = b->owningGraph();
  g->insertNode(g->createStore(name, v))->setSourceRange(loc);
  type_table[name] = std::move(type);
}

SugaredValuePtr Environment::findInThisFrame(const std::string& name) {
  auto it = value_table.find(name);
  if (it != value_table.end()) {
    return it->second;
  }
  auto it2 = type_table.find(name);
  if (it2 != type_table.end()) {
    return insertLoad(name, it2->second);
  }
  return nullptr;
}

SugaredValuePtr Environment::findInParentFrame(const std::string& name) {
  return next ? next->findInAnyFrame(name) : nullptr;
}

void Environment::setType(const std::string& name, TypePtr type) {
  type_table[name] = std::move(type);
}

SugaredValuePtr Environment::findInAnyFrame(const std::string& name) {
  for (auto runner = this; runner; runner = runner->next.get()) {
    if (auto r = runner->findInThisFrame(name)) {
      return r;
    }
  }
  return nullptr;
}

Block* Environment::block() {
  return b;
}

void Environment::setVar(
    const SourceRange& loc,
    const std::string& name,
    Value* value) {
  setSugaredVar(
      loc,
      name,
      std::make_shared<SimpleValue>(value),
      /*annotated_type=*/nullptr);
}

void Environment::setSugaredVar(
    const SourceRange& loc,
    const std::string& name,
    SugaredValuePtr value,
    TypePtr annotated_type) {
  Value* as_simple_value = asSimple(value);
  if (as_simple_value && !as_simple_value->hasDebugName() &&
      meaningfulName(name) &&
      // note: if the value wasn't defined in this block, we might be giving a
      // name only used inside this block to a value outside of this. this is
      // not normally helpful for debugging and causes import/export jitter.
      as_simple_value->node()->owningBlock() == block()) {
    as_simple_value->setDebugName(name);
  }
  // prevent re-assignment involving any sugared values
  // any reassignment like:
  // a = ...
  // while ...
  //   a = ..
  // requires 'a' to be first-class in the graph since its value depends on
  // control flow
  if (auto parent = findInParentFrame(name)) {
    if (annotated_type) {
      throw ErrorReport(loc)
          << "Attempting to declare and annotate the type of variable '" << name
          << "' but it is already defined in an outer block";
    }
    if (!as_simple_value) {
      throw ErrorReport(loc)
          << "Cannot re-assign '" << name << "' to a value of type "
          << value->kind() << " because " << name
          << " is not a first-class value.  Only reassignments to first-class values are allowed";
    }
    Value* simple_parent = asSimple(parent);
    if (!simple_parent) {
      throw ErrorReport(loc)
          << "Cannot re-assign '" << name << "' because it has type "
          << value->kind() << " and " << name
          << " is not a first-class value.  Only reassignments to first-class values are allowed";
    }

    auto parent_type = unshapedType(simple_parent->type());
    as_simple_value = tryConvertToType(
        loc,
        *b->owningGraph(),
        parent_type,
        as_simple_value,
        /*allow_conversions=*/true);
    std::stringstream why_not;
    if (!as_simple_value->type()->isSubtypeOfExt(parent_type, &why_not)) {
      auto error = ErrorReport(loc);
      error << "Variable '" << name << "' previously has type "
            << simple_parent->type()->repr_str()
            << " but is now being assigned to a value of type "
            << as_simple_value->type()->repr_str();

      // Special-cased error msg if we're trying to assign to a tensor list.
      if (simple_parent->type()->kind() == TypeKind::ListType &&
          as_simple_value->type()->kind() == TypeKind::ListType) {
        error << "\nEmpty lists default to List[Tensor]. Add a variable "
                 "annotation to the assignment to create an empty list "
                 "of another type (torch.jit.annotate(List[T, []]) where T "
                 "is the type of elements in the list for Python 2)";
      }
      error << "\n" << why_not.str();
      throw error;
    }
  }
  if (as_simple_value) {
    if (annotated_type &&
        !as_simple_value->type()->isSubtypeOf(annotated_type)) {
      throw ErrorReport(loc)
          << "Variable '" << name << "' is annotated with type "
          << annotated_type->repr_str()
          << " but is being assigned to a value of type "
          << as_simple_value->type()->repr_str();
    }
    auto value_store_type =
        annotated_type ? annotated_type : as_simple_value->type();
    insertStore(name, loc, as_simple_value, value_store_type);
  } else {
    value_table[name] = std::move(value);
  }
}

SugaredValuePtr Environment::getSugaredVar(const Ident& ident, bool required) {
  return getSugaredVar(ident.name(), ident.range());
}
Value* Environment::getVar(const Ident& ident) {
  return getSugaredVar(ident)->asValue(ident.range(), method);
}

void Environment::throwVarNotFoundError(
    const std::string& ident,
    const SourceRange& range) {
  // check if this value was not emitted in an if statement because of a
  // type mismatch. if it was, then we print a more informative error msg
  if (auto msg = findVariableTypeError(ident)) {
    throw ErrorReport(range) << *msg << "and was used here";
  }
  throw ErrorReport(range) << "undefined value " << ident;
}

SugaredValuePtr Environment::getSugaredVar(
    const std::string& ident,
    const SourceRange& range,
    bool required) {
  auto retval = findInAnyFrame(ident);

  if (!retval) {
    static std::unordered_map<std::string, SugaredValuePtr> globals = {
        {"print", std::make_shared<PrintValue>()},
        {"tuple", SpecialFormValue::create(prim::TupleConstruct)},
        {"float",
         makeMagic(
             "__float__",
             std::make_shared<CastValue>(FloatType::get(), aten::Float))},
        {"complex",
         makeMagic(
             "__complex__",
             std::make_shared<CastValue>(ComplexType::get(), aten::Complex))},
        {"int",
         makeMagic(
             "__int__",
             std::make_shared<CastValue>(IntType::get(), aten::Int))},
        {"bool",
         makeMagic(
             "__bool__",
             std::make_shared<CastValue>(BoolType::get(), aten::Bool))},
        {"str",
         makeMagic(
             "__str__",
             std::make_shared<CastValue>(StringType::get(), aten::str))},
        {"getattr", SpecialFormValue::create(prim::GetAttr)},
        {"hasattr", SpecialFormValue::create(prim::HasAttr)},
        {"isinstance", SpecialFormValue::create(prim::isinstance)},
        // todo(zach): remove when we can correctly export torch.full via ONNX
        // or we have implicit conversion that can convert numbers to tensors
        {"_to_tensor",
         std::make_shared<CastValue>(TensorType::get(), prim::NumToTensor)},
        {"len",
         makeMagic(
             "__len__",
             std::make_shared<BuiltinFunction>(aten::len, at::nullopt))},
        {"hex",
         makeMagic(
             "__hex__",
             std::make_shared<BuiltinFunction>(aten::hex, at::nullopt))},
        {"oct",
         makeMagic(
             "__oct__",
             std::make_shared<BuiltinFunction>(aten::oct, at::nullopt))},
        {"round",
         makeMagic(
             "__round__",
             std::make_shared<BuiltinFunction>(aten::round, at::nullopt))},
        {"hash", std::make_shared<BuiltinFunction>(aten::hash, at::nullopt)},
        {"id", std::make_shared<BuiltinFunction>(prim::id, at::nullopt)},
        {"min", std::make_shared<BuiltinFunction>(prim::min, at::nullopt)},
        {"max", std::make_shared<BuiltinFunction>(prim::max, at::nullopt)},
        {"abs", std::make_shared<BuiltinFunction>(prim::abs, at::nullopt)},
        {"all", std::make_shared<BuiltinFunction>(aten::all, at::nullopt)},
        {"any", std::make_shared<BuiltinFunction>(aten::any, at::nullopt)},
        {"divmod",
         std::make_shared<BuiltinFunction>(aten::divmod, at::nullopt)},
        {"sum", std::make_shared<BuiltinFunction>(aten::sum, at::nullopt)},
        {"list", SpecialFormValue::create(prim::list)},
        {"dict", SpecialFormValue::create(prim::dict)},
        {"ord", std::make_shared<BuiltinFunction>(aten::ord, at::nullopt)},
        {"chr", std::make_shared<BuiltinFunction>(aten::chr, at::nullopt)},
        {"bin", std::make_shared<BuiltinFunction>(aten::bin, at::nullopt)},
        {"pow", std::make_shared<BuiltinFunction>(aten::pow, at::nullopt)},
        {"range", SpecialFormValue::create(prim::range)},
        {"zip", SpecialFormValue::create(prim::zip)},
        {"enumerate", SpecialFormValue::create(prim::enumerate)},
        {"rangelist",
         std::make_shared<BuiltinFunction>(prim::rangelist, at::nullopt)},
        {"sorted",
         std::make_shared<BuiltinFunction>(aten::sorted, at::nullopt)},
        // Only AssertionError is bound so that we can use it from emitAssert,
        // all other exceptions should be resolved at the Python level
        {"AssertionError", std::make_shared<ExceptionValue>("AssertionError")},
    };
    auto it = globals.find(ident);
    if (it != globals.end()) {
      retval = it->second;
    }
  }

  if (!retval) {
    if (auto type = resolver->resolveType(ident, range)) {
      if (auto tuple_type = type->cast<TupleType>()) {
        retval = std::make_shared<NamedTupleConstructor>(tuple_type);
      }
    }
  }

  if (!retval) {
    retval = resolver->resolveValue(ident, method, range);
  }

  if (!retval) {
    if (auto type = resolver->resolveType(ident, range)) {
      if (auto class_type = type->cast<ClassType>()) {
        retval = std::make_shared<ClassValue>(class_type);
      }
    }
  }

  if (!retval && required) {
    throwVarNotFoundError(ident, range);
  }
  return retval;
}

Value* Environment::getVar(const std::string& ident, const SourceRange& range) {
  return getSugaredVar(ident, range)->asValue(range, method);
}

void Environment::removeVar(const Ident& ident, bool check_if_removed) {
  bool removed = false;

  for (auto runner = this; runner; runner = runner->next.get()) {
    auto a = runner->value_table.erase(ident.name());
    auto b = runner->type_table.erase(ident.name());
    removed = a || b;
  }

  if (check_if_removed && !removed) {
    throwVarNotFoundError(ident.name(), ident.range());
  }
}

std::vector<std::string> Environment::definedVariables() {
  std::vector<std::string> result;
  for (auto& kv : type_table) {
    result.push_back(kv.first);
  }
  return result;
}

} // namespace jit
} // namespace torch
