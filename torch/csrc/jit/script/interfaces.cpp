#include <torch/csrc/jit/script/interfaces.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/script/schema_matching.h>

namespace torch {
namespace jit {
namespace script {

// These schemas are not exposed in operator registeration because they have
// may have preconditions which cannot be checked in schema matching
// or because they call into other functions and can't be schematized themselves
const FunctionSchema& getInterfaceSchema(Symbol name) {
  static std::unordered_map<Symbol, const c10::FunctionSchema> schema_map = {
      {prim::sort,
       parseSchema("prim::sort(t[] self, bool reverse=False) -> ()")},
  };

  return schema_map.at(name);
}

bool matchesSortingInterface(
    const SourceRange& loc,
    c10::optional<NamedValue> self,
    Graph& graph,
    std::stringstream& failure_messages) {
  if (!self) {
    return false;
  }

  Value* self_val = self->value(graph);
  auto list_type = self_val->type()->cast<ListType>();
  if (!list_type) {
    return false;
  }

  auto element_type = list_type->getElementType()->cast<ClassType>();
  if (!element_type) {
    return false;
  }

  Function* lt_method = element_type->getMethod("__lt__");
  if (!element_type->getMethod("__lt__")) {
    failure_messages << "Could not sort list of " << element_type->str()
                     << " because it does not define a __lt__ method\n"
                     << loc;

    return false;
  };

  auto lt_schema = lt_method->getSchema();
  auto schema_args = lt_schema.arguments();
  bool error = schema_args.size() != 2;
  if (!error) {
    const auto& arg1 = schema_args[0];
    const auto& arg2 = schema_args[1];
    error |=
        (arg1.default_value() || arg1.kwarg_only() ||
         arg1.type() != element_type);
    error |=
        (arg2.default_value() || arg2.kwarg_only() ||
         arg2.type() != element_type);
  }
  error |= lt_schema.returns().size() != 1 ||
      lt_schema.returns()[0].type() != BoolType::get();
  if (error) {
    failure_messages
        << "Could not sort list of " << element_type->str()
        << " because it does not define a __lt__ method which takes in"
        << " another element of type " << element_type->str()
        << " and returns a bool\n"
        << loc;

    return false;
  }

  return true;
}

Value* tryMatchSort(
    const SourceRange& loc,
    c10::optional<NamedValue> self,
    ArrayRef<NamedValue> args,
    ArrayRef<NamedValue> kwargs,
    Graph& graph,
    std::stringstream& failure_messages,
    bool allow_conversions) {
  // check if input type can be sorted
  if (!matchesSortingInterface(loc, self, graph, failure_messages)) {
    return nullptr;
  }

  const auto sort_schema = getInterfaceSchema(prim::sort);
  const auto matched_schema = tryMatchSchema(
      sort_schema,
      loc,
      graph,
      self,
      args,
      kwargs,
      failure_messages,
      allow_conversions);

  if (matched_schema) {
    return emitBuiltinNode(*matched_schema, loc, graph, prim::sort);
  }

  return nullptr;
}

TORCH_API Value* tryMatchInterfaceOps(
    Graph& graph,
    const SourceRange& loc,
    c10::optional<NamedValue> self,
    ArrayRef<NamedValue> args,
    ArrayRef<NamedValue> kwargs,
    Symbol name,
    std::stringstream& failure_messages,
    bool allow_conversions) {
  if (name == aten::sort) {
    return tryMatchSort(
        loc, self, args, kwargs, graph, failure_messages, allow_conversions);
  }

  return nullptr;
};

} // namespace script
} // namespace jit
} // namespace torch
