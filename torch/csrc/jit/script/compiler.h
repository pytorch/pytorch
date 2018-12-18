#pragma once
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/tree_views.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/script/sugared_value.h>

namespace torch {
namespace jit {
namespace script {

using Resolver = std::function<std::shared_ptr<SugaredValue>(const std::string& name, Method& m, const SourceRange& loc)>;

inline std::shared_ptr<SugaredValue> nativeResolver(const std::string& name, Method& m, const SourceRange& loc){
  if (name == "torch") {
    return std::make_shared<BuiltinModule>("aten");
  }
  return nullptr;
}

TORCH_API void defineMethodsInModule(
  const std::shared_ptr<Module>& m,
  const std::vector<Def>& definitions,
  const std::vector<Resolver>& resolvers, /* determines how we handle free variables in each definition*/
  const std::shared_ptr<SugaredValue>& self /* if non-null, the first argument to each def, is bound to this value */
);

// same as above but parse the definitions from source
TORCH_API void defineMethodsInModule(const std::shared_ptr<Module>& m, const std::string& source, const Resolver& resolver, const std::shared_ptr<SugaredValue>& self);

TORCH_API std::vector<Value*> inlineCallTo(Graph& g, Graph& callee, ArrayRef<Value*> inputs);

// try to match a list if inputs and keyword 'attributes' to this schema,
// if it works return the flat list of positional inputs to the call
// if it returns nullopt, then failure_messages contains a good error report
// set convert_tensor_to_num to true if ImplicitTensorToNums should be inserted to
// match the schema

struct MatchedSchema {
  std::vector<Value*> inputs;
  std::vector<TypePtr> return_types;
};

TORCH_API c10::optional<MatchedSchema> tryMatchSchema(
  const FunctionSchema& schema,
  const SourceRange& loc,
  Graph& graph,
  c10::optional<NamedValue> self,
  at::ArrayRef<NamedValue> inputs,
  at::ArrayRef<NamedValue> attributes,
  std::ostream& failure_messages,
  bool allow_conversions);

TORCH_API Value* emitBuiltinCall(
  const SourceRange& loc,
  Graph& graph,
  Symbol name,
  const c10::optional<NamedValue>& self,
  at::ArrayRef<NamedValue> inputs,
  at::ArrayRef<NamedValue> attributes,
  // if true, emitBuiltinCall will throw an exception if this builtin does not exist,
  // otherwise it will return nullptr if the builtin is not found.
  bool required);

TORCH_API c10::optional<size_t> findInputWithName(
  const std::string& name,
  at::ArrayRef<NamedValue> kwargs);

TORCH_API at::ArrayRef<Value*> createTupleUnpack(Value* v);

} // namespace script
} // namespace jit
} // namespace torch
