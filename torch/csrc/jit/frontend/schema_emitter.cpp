#include <torch/csrc/jit/frontend/schema_emitter.h>

#include <torch/csrc/jit/frontend/builtin_functions.h>
#include <torch/csrc/jit/frontend/error_report.h>

#include <torch/csrc/jit/ir/named_value.h>

namespace torch {
namespace jit {

// pack outputs of a function following python rules. If there is a single value
// return a SimpleValue, otherwise pack all the values into a Tuple.
static Value* packOutputs(
    Graph& g,
    at::ArrayRef<Value*> values,
    c10::OptNameList field_names) {
  if (values.size() == 1) {
    return values[0];
  }
  std::shared_ptr<FunctionSchema> schema;
  TupleTypePtr named_tuple = nullptr;
  if (field_names) {
    auto types = fmap(values, [](Value* v) { return v->type(); });
    named_tuple =
        TupleType::createNamed(c10::nullopt, field_names.value(), types);
  }
  return g.insertNode(g.createTuple(values, named_tuple))->output();
}

MatchedSchema matchSchemaAndPrepareGraph(
    const FunctionSchema& schema,
    const SourceRange& loc,
    Graph& graph,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const c10::optional<NamedValue>& self) {
  auto matched = matchSchema(schema, loc, graph, args, kwargs, self);
  insertGraph(graph, *matched.additions, matched.inputs);
  return matched;
}

std::pair<size_t, MatchedSchema> matchSchemasAndPrepareGraph(
    std::vector<const FunctionSchema*> schemas,
    const SourceRange& loc,
    Graph& graph,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const c10::optional<NamedValue>& self) {
  auto match_pair = matchSchemas(schemas, loc, graph, args, kwargs, self);
  insertGraph(graph, *match_pair.second.additions, match_pair.second.inputs);
  return match_pair;
}

Value* tryConvertToTypeAndPrepareGraph(
    const SourceRange& loc,
    Graph& graph,
    const TypePtr& concrete_type,
    Value* value,
    bool allow_conversions) {
      auto tmp = std::shared_ptr<Graph>(new Graph());
      Value* res = tryConvertToType(loc, graph, tmp, concrete_type, value, allow_conversions);
      insertGraph(graph, *tmp, tmp->inputs());
      return res;
    }

// Given a successful match between operator schema and symbol, emit a node
// with the appropriate inputs and outputs.
Value* emitBuiltinNode(
    const MatchedSchema& matched_schema,
    const SourceRange& loc,
    Graph& graph,
    Symbol name) {
  auto n = graph.insertNode(graph.create(name, matched_schema.inputs, 0))
               ->setSourceRange(loc);

  for (auto& ret : matched_schema.return_types) {
    n->addOutput()->setType(ret);
  }

  // assert that we did indeed create an op that has implementation
  // otherwise schema and dispatch are not in sync
  n->getOperation();

  return packOutputs(graph, n->outputs(), matched_schema.return_field_names);
}

// Search for operators matching the provided symbol name and input types.
// If one is found, emit a node to the graph for that operator.
Value* emitBuiltinCall(
    const SourceRange& loc,
    Graph& graph,
    Symbol name,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const c10::optional<NamedValue>& self) {
  const auto& variants = getAllOperatorsFor(name);
  const auto& builtin_functions = getAllBuiltinFunctionsFor(name);

  std::stringstream failure_messages;
  std::vector<const FunctionSchema*> schemas;
  schemas.reserve(variants.size());
  for (const std::shared_ptr<Operator>& op : variants) {
    schemas.push_back(&op->schema());
  }
  for (const auto method : builtin_functions) {
    method->ensure_defined();
    schemas.push_back(&method->getSchema());
  }

  // no operators found with the same name, print out similarly named operators
  if (schemas.size() == 0) {
    const auto close_symbols = findSimilarOperators(name);
    auto error = ErrorReport(loc);
    const auto& user_function_name = name.toQualString();
    error << "Unknown builtin op: " << user_function_name << ".\n";
    if (close_symbols.size() == 0) {
      error
          << "Could not find any similar ops to " << user_function_name
          << ". This op may not exist or may not be currently supported in TorchScript.\n";
    } else {
      error << "Here are some suggestions: \n";
      for (const auto& sym : close_symbols) {
        error << "\t" << sym.toQualString() << "\n";
      }
      error << "\nThe original call is";
    }
    throw error;
  }

  auto matched = matchSchemasAndPrepareGraph(
      schemas, loc, graph, args, kwargs, self);

  if (matched.first < variants.size()) {
    return emitBuiltinNode(matched.second, loc, graph, name);
  } else {
    Function* fn = builtin_functions[matched.first - variants.size()];
    // we inline builtin calls because they are normally very small
    // wrappers and are not useful for keeping around to debug
    return insertGraph(graph, *fn->graph(), matched.second.inputs).at(0);
  }
}

} // namespace jit
} // namespace torch
