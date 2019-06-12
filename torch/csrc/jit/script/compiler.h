#pragma once
#include <functional>
#include <memory>
#include <string>

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/script/error_report.h"
#include "torch/csrc/jit/script/tree_views.h"
#include "torch/csrc/jit/script/module.h"

namespace torch {
namespace jit {
namespace script {

static inline std::vector<Value*> toValues(Graph& g, at::ArrayRef<NamedValue> nvs) {
  return fmap(nvs, [&](const NamedValue& v) {
    return v.value(g);
  });
}

// The AST can contain nodes like `self`, `self.b` or `python_fn` that
// are not first-class values in the graph representation, but instead
// will be desugared based on how they are used in the AST.

// SugaredValue is used to temporarily represent these values in a way
// that separates their behavior from the AST -> IR converter itself.
// This allows us to keep dependencies on python minimal.

struct SugaredValue : public std::enable_shared_from_this<SugaredValue> {
  // what is this node? for error reporting (e.g. Module, python function)
  virtual std::string kind() const = 0;

  // what can we do with this thing?
  // use it as a value e.g.  `this + 4`
  virtual Value * asValue(SourceRange loc, Method & m) {
    throw ErrorReport(loc) << kind() << " cannot be used as a value";
  }

  // select an attribute on it, e.g. `this.field`
  virtual std::shared_ptr<SugaredValue> attr(SourceRange loc, Method & m, const std::string& field) {
    throw ErrorReport(loc) << "attribute lookup is not defined on " << kind();
  }

  // use it as a vector of values, e.g. a tuple of values as return value from
  // a method invocation
  virtual std::vector<std::shared_ptr<SugaredValue>> asTuple(
      SourceRange loc,
      Method& m,
      c10::optional<size_t> size_hint = {}) {
    throw ErrorReport(loc) << kind() << " cannot be used as a tuple";
  }

  // call it like a function, e.g. `outputs = this(inputs)`
  virtual std::shared_ptr<SugaredValue> call(
    SourceRange loc,
    Method & m,
    // note: names for args will be 'argument 0', 'argument 1', etc..
    at::ArrayRef<NamedValue> inputs_,
    at::ArrayRef<NamedValue> attributes,
    size_t n_binders) {
// n_binders is always set to the number of variables an expression is
// syntactically bound to:
//     a = foo() # 1 binder (note in this case the single binder might be a tuple)
//     a, * b = foo() # 1 binder
//     a, b = foo() # 2 binders
//     foo() # 0 binders
//
// In subexpressions, like bar() in foo(bar()), n_binders is always set to
// 1. n_binders is used as a hint to subexpressions to determine how many
// values they should return when that number is ambiguous statically. In
// particular it is currently used to decide how many tensors a call to a
// python function will return. It is only a hint, functions do not have to
// check that n_binders match the number of things they are returning, the
// assignment logic will do that anyway.

    throw ErrorReport(loc) << "cannot call a " << kind();
  }

  virtual ~SugaredValue() = default;
};

// most things in the environment are just simple value types
// and not special python syntax sugar types
struct TORCH_API SimpleValue : public SugaredValue {
  SimpleValue(Value * value)
  : value(value) {}
  std::string kind() const override {
    return "value";
  }
  Value * asValue(SourceRange range, Method & m) override {
    return value;
  }
  std::vector<std::shared_ptr<SugaredValue>> asTuple(
      SourceRange loc,
      Method& m,
      c10::optional<size_t> size_hint = {}) override;
  std::shared_ptr<SugaredValue> attr(SourceRange loc, Method & m, const std::string& field) override;
  Value* getValue() const {
    return value;
  }
private:
  Value* value;
};

struct TORCH_API BuiltinFunction : public SugaredValue {
  BuiltinFunction(Symbol symbol, c10::optional<NamedValue> self)
      : symbol(std::move(symbol)), self(std::move(self)) {}

  // The symbol of the function (e.g. `aten::relu`).
  Symbol symbol;

  // if this is method, then this is the self argument.
  c10::optional<NamedValue> self;

  std::string kind() const override {
    return "builtin";
  }
  std::shared_ptr<SugaredValue> call(
      SourceRange loc,
      Method& m,
      at::ArrayRef<NamedValue> attributes,
      at::ArrayRef<NamedValue> inputs,
      size_t n_binders) override;
};

struct TORCH_API BuiltinModule : public SugaredValue {
  BuiltinModule(std::string name,
                c10::optional<int64_t> version = at::nullopt)
    : name(std::move(name))
    , version(version) {}

  std::string kind() const override {
    return "builtin module";
  }
  std::shared_ptr<SugaredValue> attr(SourceRange loc, Method & m, const std::string& field) override {
    return std::make_shared<BuiltinFunction>(Symbol::fromQualString(name+"::"+field), c10::nullopt);
  }

private:
  std::string name;
  // when we add operator versioning, emit this op as it exising at 'version'
  // if not set, use the latest version
  c10::optional<int64_t> version;
};

// These SugaredValues have special handling in the compiler because they
// change the normal evalution order of the expression they participate in.
// They are exposed here so that the python frontend can inject them
// when it sees the equivalent thing in python
struct TORCH_API ForkValue : public SugaredValue {
  ForkValue() = default;
  std::string kind() const override {
    return "fork";
  }
};
struct TORCH_API AnnotateValue : public SugaredValue {
  AnnotateValue() = default;
  std::string kind() const override {
    return "annotate";
  }
};

using Resolver = std::function<std::shared_ptr<SugaredValue>(const std::string& name, Method& m, const SourceRange& loc)>;

inline std::shared_ptr<SugaredValue> nativeResolver(const std::string& name, Method& m, const SourceRange& loc){
  if (name == "torch") {
    return std::make_shared<BuiltinModule>("aten");
  }
  return nullptr;
}

TORCH_API void defineMethodsInModule(
  Module & m,
  const std::vector<Def>& definitions,
  const std::vector<Resolver>& resolvers, /* determines how we handle free variables in each definition*/
  std::shared_ptr<SugaredValue> self /* if non-null, the first argument to each def, is bound to this value */
);

// same as above but parse the definitions from source
TORCH_API void defineMethodsInModule(Module & m, const std::string& source, Resolver resolver, std::shared_ptr<SugaredValue> self);

// pack outputs of a function following python rules. If there is a single value return
// a SimpleValue, otherwise pack all the values into a Tuple.
TORCH_API Value* packOutputs(Graph& g, at::ArrayRef<Value*> values);
TORCH_API std::vector<Value*> inlineCallTo(Graph& g, Graph& callee, ArrayRef<Value*> inputs);

// defines how a method obtained from a module behaves in script
struct MethodValue : public SugaredValue {
  MethodValue(std::shared_ptr<Module> module, Method& method)
  : module(std::move(module)) //insurance that method stays alive
  , method(method) {}
  std::string kind() const override {
    return "method";
  }
  std::shared_ptr<SugaredValue> call(SourceRange loc, Method & caller, at::ArrayRef<NamedValue> inputs, at::ArrayRef<NamedValue> attributes, size_t n_binders) override {
    return std::make_shared<SimpleValue>(packOutputs(*caller.graph(), caller.emit_call_to(loc, method, inputs, attributes)));
  }
private:
  std::shared_ptr<Module> module;
  Method& method;

};

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
  bool convert_tensors_to_nums);

TORCH_API Value* emitBuiltinCall(
  const SourceRange& loc,
  Graph& graph,
  Symbol name,
  c10::optional<NamedValue> self,
  at::ArrayRef<NamedValue> inputs,
  at::ArrayRef<NamedValue> attributes,
  // if true, emitBuiltinCall will throw an exception if this builtin does not exist,
  // otherwise it will return nullptr if the builtin is not found.
  bool required);

} // namespace script
} // namespace jit
} // namespace torch
