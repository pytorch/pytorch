#pragma once
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/script/tree_views.h>

namespace torch {
namespace jit {
namespace script {

// The AST can contain nodes like `self`, `self.b` or `python_fn` that
// are not first-class values in the graph representation, but instead
// will be desugared based on how they are used in the AST.

// SugaredValue is used to temporarily represent these values in a way
// that separates their behavior from the AST -> IR converter itself.
// This allows us to keep dependencies on python minimal.

enum NoneStatus { ALWAYS, MAYBE, NEVER };

struct TORCH_API SugaredValue : public std::enable_shared_from_this<SugaredValue> {
  // what is this node? for error reporting (e.g. Module, python function)
  virtual std::string kind() const = 0;

  // what can we do with this thing?
  // use it as a value e.g.  `this + 4`
  virtual Value* asValue(const SourceRange& loc, Function& m) {
    throw ErrorReport(loc) << kind() << " cannot be used as a value";
  }

  // select an attribute on it, e.g. `this.field`
  virtual std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) {
    throw ErrorReport(loc) << "attribute lookup is not defined on " << kind();
  }

  // assign an attribute on it, e.g. `this.field = newValue`
  virtual void setAttr(
      const SourceRange& loc,
      Function& m,
      const std::string& field,
      Value* newValue) {
    throw ErrorReport(loc) << "attribute assignment is not defined on "
                           << kind();
  }
  virtual NoneStatus isNone() {
    return NEVER;
  }

  // use it as a vector of values, e.g. a tuple of values as return value from
  // a method invocation
  virtual std::vector<std::shared_ptr<SugaredValue>> asTuple(
      const SourceRange& loc,
      Function& m,
      const c10::optional<size_t>& size_hint = {}) {
    throw ErrorReport(loc) << kind() << " cannot be used as a tuple";
  }

  virtual std::vector<std::shared_ptr<SugaredValue>> asType(
      const SourceRange& loc,
      Method& m) {
    throw ErrorReport(loc) << kind() << " cannot be used as a type";
  }

  // call it like a function, e.g. `outputs = this(inputs)`
  virtual std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      // note: names for args will be 'argument 0', 'argument 1', etc..
      at::ArrayRef<NamedValue> inputs_,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) {
    // n_binders is always set to the number of variables an expression is
    // syntactically bound to:
    //     a = foo() # 1 binder (note in this case the single binder might be a
    //     tuple) a, * b = foo() # 1 binder a, b = foo() # 2 binders foo() # 0
    //     binders
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
  SimpleValue(Value* value) : value_(value) {}
  std::string kind() const override {
    return "value";
  }
  Value* asValue(const SourceRange& range, Function& m) override {
    return value_;
  }
  NoneStatus isNone() override {
    if (value_->mustBeNone())
      return ALWAYS;
    else if (value_->type()->cast<OptionalType>())
      return MAYBE;
    else
      return NEVER;
  }
  std::vector<std::shared_ptr<SugaredValue>> asTuple(
      const SourceRange& loc,
      Function& m,
      const c10::optional<size_t>& size_hint = {}) override;
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) override;

  void setAttr(
      const SourceRange& loc,
      Function& m,
      const std::string& field,
      Value* newValue) override;

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      // note: names for args will be 'argument 0', 'argument 1', etc..
      at::ArrayRef<NamedValue> inputs_,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override;

  Value* getValue() const {
    return value_;
  }

 private:
  Value* value_;
};

struct TORCH_API BuiltinFunction : public SugaredValue {
  BuiltinFunction(Symbol symbol, c10::optional<NamedValue> self)
      : symbol(symbol), self(std::move(self)) {}

  // The symbol of the function (e.g. `aten::relu`).
  Symbol symbol;

  // if this is method, then this is the self argument.
  c10::optional<NamedValue> self;

  std::string kind() const override {
    return "builtin";
  }
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      at::ArrayRef<NamedValue> attributes,
      at::ArrayRef<NamedValue> inputs,
      size_t n_binders) override;
};

struct TORCH_API BuiltinModule : public SugaredValue {
  BuiltinModule(std::string name, c10::optional<int64_t> version = at::nullopt)
      : name(std::move(name)), version(std::move(version)) {}

  std::string kind() const override {
    return "builtin module";
  }
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) override {
    return std::make_shared<BuiltinFunction>(
        Symbol::fromQualString(name + "::" + field), c10::nullopt);
  }

 private:
  std::string name;
  // when we add operator versioning, emit this op as it exising at 'version'
  // if not set, use the latest version
  c10::optional<int64_t> version;
};

// Represents a class, analagous to `int` or `dict`. Instances of classes,
// like `1` or `{"foo": 5}`, are represented as SimpleValues
struct TORCH_API ClassValue : public SugaredValue {
  explicit ClassValue(ClassTypePtr type) : type_(std::move(type)) {}

  // Call the type's constructor, as in:
  //    n = Foo(constructor_arg)
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override;

  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) override;

  std::string kind() const override {
    return type_->str();
  }

  ClassTypePtr type_;
};

// defines how a method obtained from a module behaves in script
struct MethodValue : public SugaredValue {
  MethodValue(c10::optional<NamedValue> self, std::shared_ptr<Function> method)
      : self_(std::move(self)), method_(std::move(method)) {}
  std::string kind() const override {
    return "method";
  }
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& f,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override {
    Graph& graph = *f.graph();
    if (self_) {
      std::vector<NamedValue> inputsWithSelf;
      inputsWithSelf.emplace_back(loc, self_->value(graph));
      inputsWithSelf.insert(inputsWithSelf.end(), inputs.begin(), inputs.end());
      return std::make_shared<SimpleValue>(
          method_->emit_call(graph, loc, inputsWithSelf, attributes));
    }

    return std::make_shared<SimpleValue>(
        method_->emit_call(graph, loc, inputs, attributes));
  }

 private:
  c10::optional<NamedValue> self_;
  std::shared_ptr<Function> method_;
};

struct TORCH_API PrintValue : public SugaredValue {
  std::string kind() const override {
    return "print";
  }
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override;
};

// expressions like int(x)
// these are the same as call prim::Int or equivalent except it
// is a noop when the input is a subtype of 'type'
struct TORCH_API CastValue : public BuiltinFunction {
  CastValue(TypePtr type, c10::Symbol method)
      : BuiltinFunction(method, c10::nullopt), type_(std::move(type)) {}
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override {
    if (inputs.size() == 1 && attributes.size() == 0) {
      auto v = inputs[0].value(*m.graph());
      if (v->type()->isSubtypeOf(type_)) {
        return std::make_shared<SimpleValue>(v);
      }
    }
    return BuiltinFunction::call(loc, m, inputs, attributes, n_binders);
  }

 private:
  TypePtr type_;
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

// matched against for special handling of getattr expressions
struct TORCH_API GetAttrValue : SugaredValue {
  GetAttrValue() = default;
  std::string kind() const override {
    return "getattr";
  }
};

// matched against for special handling of isinstance expressions
struct TORCH_API IsInstanceValue : SugaredValue {
  IsInstanceValue() = default;
  std::string kind() const override {
    return "isinstance";
  }
};

// This represents the "__new__" method on classes, which can't be a MethodValue
// because it takes a ClassValue as input.
// So if we see:
//   Foo.__new__(Foo)
// Foo is a ClassValue, calling `attr("__new__")` will return a ClassNewMethod.
struct TORCH_API ClassNewMethod : public SugaredValue {
  ClassNewMethod(ClassTypePtr type) : type_(type) {}
  std::string kind() const override {
    return "class.__new__";
  }

  std::shared_ptr<SugaredValue> createObject(
      const SourceRange& loc,
      Function& m) {
    auto& g = *m.graph();
    auto createNode = g.insertNode(g.createObject(type_));
    return std::make_shared<SimpleValue>(createNode->output());
  }

  ClassTypePtr type_;
};

static inline std::vector<Value*> toValues(
    Graph& g,
    at::ArrayRef<NamedValue> nvs) {
  return fmap(nvs, [&](const NamedValue& v) { return v.value(g); });
}

static inline Self simpleSelf(const TypePtr& typ) {
  return [typ](Value* v) {
    v->setType(typ);
    return std::make_shared<SimpleValue>(v);
  };
}
} // namespace script
} // namespace jit
} // namespace torch
