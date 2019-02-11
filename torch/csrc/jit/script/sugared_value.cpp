#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/sugared_value.h>
#include <torch/csrc/jit/script/tree_views.h>
#include <torch/csrc/jit/script/type_parser.h>

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

// support syntax sugar for x.foo(y, z) by allowing x.foo to return a
// callable value that will resolve to foo(x, y, z) when called.
std::shared_ptr<SugaredValue> SimpleValue::attr(
    const SourceRange& loc,
    Method& m,
    const std::string& field) {
  // Allow method-style casts on Tensor types. e.g. x.int()
  if (value->type()->isSubtypeOf(TensorType::get())) {
    if (builtin_cast_methods().count(field)) {
      return std::make_shared<BuiltinFunction>(
          Symbol::aten(builtin_cast_methods().at(field)),
          NamedValue(loc, "self", value));
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
          m.graph()->insert(Symbol::fromQualString("prim::" + field), {value});
      return std::make_shared<SimpleValue>(r);
    }
  }
  if (getValue()->type()->isSubtypeOf(NumberType::get())) {
    throw ErrorReport(loc) << "Cannot call methods on numbers";
  }
  if (getValue()->type()->kind() == TypeKind::TupleType) {
    auto tuple_type = getValue()->type()->expect<TupleType>();
    if (!tuple_type->hasNames()) {
      throw ErrorReport(loc) << "Getting attributes of tuples is not supported";
    }
    auto names = tuple_type->names();
    for (int i = 0; i < names.size(); i++) {
      if (names[i] == field) {
        auto r = m.graph()->insertNode(m.graph()->createTupleIndex(getValue(), i))->output();
        return std::make_shared<SimpleValue>(r);
      }
    }
    throw ErrorReport(loc) << "Unknown attribute to named tuple";
  }
  return std::make_shared<BuiltinFunction>(
      Symbol::aten(field), NamedValue(loc, "self", value));
}

std::vector<std::shared_ptr<SugaredValue>> SimpleValue::asTuple(
    const SourceRange& loc,
    Method& m,
    const c10::optional<size_t>& size_hint) {
  static const auto make_simple_value =
      [](Value* v) -> std::shared_ptr<SugaredValue> {
    return std::make_shared<SimpleValue>(v);
  };
  if (value->type()->kind() == TypeKind::TupleType) {
    auto outputs = createTupleUnpack(value);
    return fmap(outputs, make_simple_value);
  } else if (value->type()->kind() == TypeKind::ListType) {
    if (!size_hint) {
      throw ErrorReport(loc)
          << "cannot statically infer the expected size of a list in this context";
    }
    auto graph = value->owningGraph();
    Node* unpack =
        graph->insertNode(graph->createListUnpack(value, *size_hint));
    return fmap(unpack->outputs(), make_simple_value);
  }
  throw ErrorReport(loc) << value->type()->str()
                         << " cannot be used as a tuple";
}

} // namespace script
} // namespace jit
} // namespace torch
