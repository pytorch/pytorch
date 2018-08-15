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

struct CallsiteDescriptor {
  size_t n_outputs;
  bool allow_varargs;
};

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
  virtual std::vector<std::shared_ptr<SugaredValue>> asTuple(SourceRange loc, Method& m) {
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
  virtual std::string kind() const override {
    return "value";
  }
  virtual Value * asValue(SourceRange range, Method & m) override {
    return value;
  }
  virtual std::vector<std::shared_ptr<SugaredValue>> asTuple(SourceRange loc, Method& m) override;
  virtual std::shared_ptr<SugaredValue> attr(SourceRange loc, Method & m, const std::string& field) override;
  Value* getValue() const {
    return value;
  }
private:
  Value* value;
};

struct TORCH_API BuiltinFunction : public SugaredValue {
  BuiltinFunction(const std::string& name, at::optional<NamedValue> value)
    : name(name), value(std::move(value)) {}
  std::string name;

  // if this is method, then this is the self argument.
  at::optional<NamedValue> value;

  virtual std::string kind() const override {
    return "builtin";
  }
  virtual std::shared_ptr<SugaredValue> call(
    SourceRange loc,
    Method & m,
    at::ArrayRef<NamedValue> attributes,
    at::ArrayRef<NamedValue> inputs,
    size_t n_binders) override;
};

using Resolver = std::function<std::shared_ptr<
    SugaredValue>(const std::string& name, Method& m, const SourceRange& loc)>;
TORCH_API void defineMethodsInModule(
  Module & m,
  const std::vector<Def>& definitions,
  const std::vector<Resolver>& resolvers, /* determines how we handle free variables in each definition*/
  std::shared_ptr<SugaredValue> self /* if non-null, the first argument to each def, is bound to this value */
);

// same as above but parse the definitions from source
TORCH_API void defineMethodsInModule(Module & m, const std::string& source, const Resolver& resolver, std::shared_ptr<SugaredValue> self);
TORCH_API std::shared_ptr<Graph> compileFunction(Def def, const Resolver& resolver);

// pack outputs of a function following python rules. If there is a single value return
// a SimpleValue, otherwise pack all the values into a Tuple.
TORCH_API Value* packOutputs(Graph& g, at::ArrayRef<Value*> values);
TORCH_API std::vector<Value*> inlineCallTo(Graph& g, Graph& callee, ArrayRef<Value*> inputs);
TORCH_API void ensureSizeMatches(SourceRange loc, size_t expected, size_t actual, const std::string& what);
TORCH_API void ensureTensors(const SourceRange& range, at::ArrayRef<Value*> values);

// try to match a list if inputs and keyword 'attributes' to this schema,
// if it works return the flat list of positional inputs to the call
// if it returns nullopt, then failure_messages contains a good error report
TORCH_API at::optional<std::vector<Value*>> tryMatchSchema(
  const FunctionSchema& schema,
  const SourceRange& loc,
  Graph& graph,
  at::ArrayRef<NamedValue> inputs,
  at::ArrayRef<NamedValue> attributes,
  std::ostream& failure_messages);

TORCH_API FunctionSchema extractSchemaFromDef(const Def &def, bool is_method=false);

TORCH_API Value* emitBuiltinCall(
  const SourceRange& loc,
  Graph& graph,
  Symbol name,
  at::ArrayRef<NamedValue> inputs,
  at::ArrayRef<NamedValue> attributes,
  // if true, emitBuiltinCall will throw an exception if this builtin does not exist,
  // otherwise it will return nullptr if the builtin is not found.
  bool required);

} // namespace script
} // namespace jit
} // namespace torch
