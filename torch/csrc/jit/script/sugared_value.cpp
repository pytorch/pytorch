#include <torch/csrc/jit/script/sugared_value.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/schema_matching.h>
#include <torch/csrc/jit/script/tree_views.h>

namespace torch {
namespace jit {
namespace script {

struct NoneValue : SugaredValue {
  NoneValue() = default;
  std::string kind() const override {
    return "None";
  }
};

std::shared_ptr<SugaredValue> PrintValue::call(
    const SourceRange& loc,
    Method& m,
    at::ArrayRef<NamedValue> inputs,
    at::ArrayRef<NamedValue> attributes,
    size_t n_binders) {
  auto& g = *m.graph();
  if (!attributes.empty())
    throw ErrorReport(loc) << "print doesn't accept any keyword arguments";

  // temporary hack to allow print statements to work in python 2, where
  // print(a, b) is treated as a (a, b) tuple input.

  std::vector<Value*> lowered_inputs = toValues(*m.graph(), inputs);
  if (lowered_inputs.size() == 1 &&
      lowered_inputs.at(0)->node()->kind() == prim::TupleConstruct) {
    auto input = lowered_inputs[0];
    for (size_t j = 0; j < input->node()->inputs().size(); ++j) {
      lowered_inputs.insert(
          lowered_inputs.begin() + 1 + j, input->node()->inputs().at(j));
    }
    lowered_inputs.erase(lowered_inputs.begin());
  }
  g.insertNode(g.create(prim::Print, lowered_inputs, 0)
                   ->setSourceLocation(std::make_shared<SourceRange>(loc)));
  return std::make_shared<NoneValue>();
}

static const std::unordered_map<std::string, std::string>&
builtin_cast_methods() {
  static std::unordered_map<std::string, std::string> builtin_cast_methods = {
      {"byte", "_cast_Byte"},
      {"char", "_cast_Char"},
      {"double", "_cast_Double"},
      {"float", "_cast_Float"},
      {"int", "_cast_Int"},
      {"long", "_cast_Long"},
      {"short", "_cast_Short"},
      {"half", "_cast_Half"}};
  return builtin_cast_methods;
}

std::shared_ptr<SugaredValue> BuiltinFunction::call(
    const SourceRange& loc,
    Method& m,
    at::ArrayRef<NamedValue> inputs,
    at::ArrayRef<NamedValue> attributes,
    size_t n_binders) {
  return std::make_shared<SimpleValue>(
      emitBuiltinCall(loc, *m.graph(), symbol, self, inputs, attributes, true));
}

// support syntax sugar for x.foo(y, z) by allowing x.foo to return a
// callable value that will resolve to foo(x, y, z) when called.
std::shared_ptr<SugaredValue> SimpleValue::attr(
    const SourceRange& loc,
    Method& m,
    const std::string& field) {
  // Allow method-style casts on Tensor types. e.g. x.int()
  if (value_->type()->isSubtypeOf(TensorType::get())) {
    if (builtin_cast_methods().count(field)) {
      return std::make_shared<BuiltinFunction>(
          Symbol::aten(builtin_cast_methods().at(field)),
          NamedValue(loc, "self", value_));
    }
    // functions that are just direct property lookups on tensor
    // must be registered as prim::<name>(Tensor t) -> <return_type>
    static const std::unordered_set<std::string> fields = {
        "dtype",
        "device",
        "shape",
        "is_cuda",
        "requires_grad",
    };
    if (fields.count(field)) {
      auto r =
          m.graph()->insert(Symbol::fromQualString("prim::" + field), {value_});
      return std::make_shared<SimpleValue>(r);
    }
  }
  if (value_->type()->isSubtypeOf(NumberType::get())) {
    throw ErrorReport(loc) << "Cannot call methods on numbers";
  }
  if (auto tuple_type = value_->type()->cast<TupleType>()) {
    if (!tuple_type->hasNames()) {
      throw ErrorReport(loc) << "Getting attributes of tuples is not supported";
    }
    auto names = tuple_type->names();
    for (size_t i = 0; i < names.size(); i++) {
      if (names[i] == field) {
        auto r = m.graph()
                     ->insertNode(m.graph()->createTupleIndex(value_, i))
                     ->output();
        return std::make_shared<SimpleValue>(r);
      }
    }
    throw ErrorReport(loc) << "Unknown attribute to named tuple";
  }

  if (auto userType = value_->type()->cast<UserType>()) {
    // This is a user-defined type, emit the proper attribute lookup
    if (auto method = userType->getMethod(field)) {
      return std::make_shared<MethodValue>(shared_from_this(), *method);
    }

    if (!userType->hasAttribute(field)) {
      throw ErrorReport(loc)
          << "Tried to access to nonexistent attribute " << field
          << ". Did you forget to initialize it in __init__()?";
    }
    auto& g = *m.graph();
    auto n = g.insertNode(g.createGetAttr(value_, field));
    return std::make_shared<SimpleValue>(n->output());
  }

  return std::make_shared<BuiltinFunction>(
      Symbol::aten(field), NamedValue(loc, "self", value_));
}

std::vector<std::shared_ptr<SugaredValue>> SimpleValue::asTuple(
    const SourceRange& loc,
    Method& m,
    const c10::optional<size_t>& size_hint) {
  static const auto make_simple_value =
      [](Value* v) -> std::shared_ptr<SugaredValue> {
    return std::make_shared<SimpleValue>(v);
  };
  if (value_->type()->kind() == TypeKind::TupleType) {
    auto outputs = createTupleUnpack(value_);
    return fmap(outputs, make_simple_value);
  } else if (value_->type()->kind() == TypeKind::ListType) {
    if (!size_hint) {
      throw ErrorReport(loc)
          << "cannot statically infer the expected size of a list in this context";
    }
    auto graph = value_->owningGraph();
    Node* unpack =
        graph->insertNode(graph->createListUnpack(value_, *size_hint));
    return fmap(unpack->outputs(), make_simple_value);
  }
  throw ErrorReport(loc) << value_->type()->str()
                         << " cannot be used as a tuple";
}

void SimpleValue::setAttr(
    const SourceRange& loc,
    Method& m,
    const std::string& field,
    Value* newValue,
    bool shouldDefine) {
  const auto userType = value_->type()->cast<UserType>();
  if (!userType) {
    throw ErrorReport(loc) << "Tried to set an attribute: " << field
                           << " on a non-user-defined type: "
                           << value_->type()->str();
  }

  auto expectedType = userType->getAttribute(field);
  if (!expectedType) {
    // We don't have an attribute with this name, either add it to the type
    // definition or throw an error
    if (shouldDefine) {
      userType->addAttribute(field, newValue->type());
      expectedType = newValue->type();
      const auto insertPoint = m.graph()->insertPoint();
      const auto topLevelBlock = m.graph()->block();
      if (insertPoint->owningBlock() != topLevelBlock) {
        throw ErrorReport(loc)
            << "First assignment cannot be in a control-flow block. "
            << "Initialize the field at the top level first.";
      }
    } else {
      throw ErrorReport(loc)
          << "Tried to set nonexistent attribute: " << field
          << ". Did you forget to initialize it in __init__()?";
    }
  }

  // Check type correctness
  const auto newType = newValue->type();
  if (!newType->isSubtypeOf(expectedType)) {
    throw ErrorReport(loc) << "Wrong type for attribute assignment. Expected "
                           << expectedType->str() << " but got "
                           << newType->str();
  }

  auto& g = *m.graph();
  g.insertNode(g.createSetAttr(value_, field, newValue));
}

std::shared_ptr<SugaredValue> UserTypeValue::call(
    const SourceRange& loc,
    Method& m,
    // note: names for args will be 'argument 0', 'argument 1', etc..
    at::ArrayRef<NamedValue> inputs,
    at::ArrayRef<NamedValue> attributes,
    size_t n_binders) {
  AT_ASSERT(n_binders <= 1);

  // Generate a new object of the right type, then call `__init__` on it
  auto& g = *m.graph();
  auto createNode = g.insertNode(g.createUserObject(type_));
  auto self = std::make_shared<SimpleValue>(createNode->output());

  auto initMethod = type_->getMethod("__init__");
  AT_ASSERT(initMethod);

  // Call the init function
  MethodValue(self, *initMethod).call(loc, m, inputs, attributes, n_binders);

  return self;
}
} // namespace script
} // namespace jit
} // namespace torch
