#pragma once
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/script/schema_matching.h>

namespace torch {
namespace jit {
namespace script {

// The AST can contain nodes like `self`, `self.b` or `python_fn` that
// are not first-class values in the graph representation, but instead
// will be desugared based on how they are used in the AST.

// SugaredValue is used to temporarily represent these values in a way
// that separates their behavior from the AST -> IR converter itself.
// This allows us to keep dependencies on python minimal.

struct TORCH_API SugaredValue
    : public std::enable_shared_from_this<SugaredValue> {
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

  // return length of this thing, if not then it can't be iterated.
  virtual Value* len(const SourceRange& loc, Function& m) {
    throw ErrorReport(loc) << "'" << kind() << "'"
                           << " object is not iterable";
  }
  // expression for ith elemement for iterable value
  virtual Value* getitem(const SourceRange& loc, Function& m, Value* idx) {
    throw ErrorReport(loc) << "'" << kind() << "'"
                           << " object is not subscriptable";
  }

  virtual ~SugaredValue() = default;
};

// most things in the environment are just simple value types
// and not special python syntax sugar types
struct TORCH_API SimpleValue : public SugaredValue {
  SimpleValue(Value* value) : value_(value) {}
  std::string kind() const override {
    std::stringstream ss;
    ss << "value of type '" << value_->type()->python_str() << "'";
    return ss.str();
  }
  Value* asValue(const SourceRange& range, Function& m) override {
    return value_;
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

  Value* len(const SourceRange& loc, Function& m) override;
  Value* getitem(const SourceRange& loc, Function& m, Value* idx) override;

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

struct TORCH_API NamedTupleConstructor : public SugaredValue {
  explicit NamedTupleConstructor(TupleTypePtr type) : type_(std::move(type)) {}

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override;

  std::string kind() const override {
    return type_->str();
  }

  TupleTypePtr type_;
};

struct FunctionValue : public SugaredValue {
  FunctionValue(Function* callee) : callees_({std::move(callee)}) {}
  FunctionValue(const StrongFunctionPtr& p)
      : callees_({p.function_}), cu_(p.cu_) {}
  FunctionValue(const std::vector<StrongFunctionPtr>& callees) {
    for (const StrongFunctionPtr& callee : callees) {
      cu_ = cu_ ? cu_ : callee.cu_;
      TORCH_INTERNAL_ASSERT(callee.cu_ == cu_);
      callees_.push_back(callee.function_);
    }
  }

  std::string kind() const override {
    return "function";
  }

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& f,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override {
    std::vector<const FunctionSchema*> schemas;
    for (Function* callee : callees_) {
      callee->ensure_defined();
      schemas.push_back(&callee->getSchema());
    }
    auto match = matchSchemas(schemas, loc, *f.graph(), inputs, attributes);
    Value* output =
        f.graph()->insertFunctionCall(callees_[match.first], match.second);
    output->node()->setSourceRange(loc);
    return std::make_shared<SimpleValue>(output);
  }

 private:
  std::vector<Function*> callees_;
  // TODO holding this thing is creepy
  std::shared_ptr<CompilationUnit> cu_;
};

struct TORCH_API ClosureValue : public SugaredValue {
  ClosureValue(Value* value) : value_(value) {
    TORCH_INTERNAL_ASSERT(value_->node()->kind() == prim::Function);
  }
  std::string kind() const override {
    return "closure";
  }
  Value* asValue(const SourceRange& range, Function& m) override {
    return value_;
  }
  Value* value_;
};

// defines how a method obtained from a module behaves in script
struct MethodValue : public SugaredValue {
  MethodValue(Value* self, std::vector<std::string> method_names)
      : self_(std::move(self)), method_names_(std::move(method_names)) {}
  MethodValue(Value* self, std::string method_name)
      : MethodValue(self, std::vector<std::string>({method_name})) {}

  std::string kind() const override {
    return "method";
  }

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& f,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override {
    std::vector<NamedValue> inputsWithSelf = {self_};
    inputsWithSelf.insert(inputsWithSelf.end(), inputs.begin(), inputs.end());
    std::vector<const FunctionSchema*> schemas;
    for (const std::string& method_name : method_names_) {
      if (auto class_type = self_->type()->cast<ClassType>()) {
        auto method = class_type->getMethod(method_name);
        TORCH_INTERNAL_ASSERT(method);
        method->ensure_defined();
        schemas.push_back(&method->getSchema());
      } else if (auto interface_type = self_->type()->cast<InterfaceType>()) {
        schemas.push_back(interface_type->getMethod(method_name));
      } else {
        TORCH_INTERNAL_ASSERT(
            false, "method constructed that is not a class or interface");
      }
    }
    auto match =
        matchSchemas(schemas, loc, *f.graph(), inputsWithSelf, attributes);
    Value* output =
        f.graph()->insertMethodCall(method_names_[match.first], match.second);
    output->node()->setSourceRange(loc);
    return std::make_shared<SimpleValue>(output);
  }

 private:
  Value* self_;
  std::vector<std::string> method_names_;
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

using SugaredValuePtr = std::shared_ptr<SugaredValue>;

// builtins operators and functions that call a method if it exists
// on a class type, like 'len(x)' and 'x + y'
struct TORCH_API MagicMethod : public SugaredValue {
  MagicMethod(std::string desugared_name, SugaredValuePtr base)
      : base_value_(std::move(base)),
        desugared_name_(std::move(desugared_name)) {}

  std::string kind() const override {
    return desugared_name_;
  }

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override;

 private:
  SugaredValuePtr base_value_;
  std::string desugared_name_;
};

// things that look like function applications, but
// perform non-standard evaluation are represented
// with SpecialFormValues, e.g.
//   isinstance(x, int)
//   fork(fn)
//   annotate(int, 3)
// The implementation of each value is handled by a case inside emitApplyExpr
struct TORCH_API SpecialFormValue : public SugaredValue {
  SpecialFormValue(Symbol form) : form_(form) {}
  std::string kind() const override {
    return form_.toUnqualString();
  }
  Symbol form() const {
    return form_;
  }
  static std::shared_ptr<SpecialFormValue> create(Symbol form) {
    return std::make_shared<SpecialFormValue>(form);
  }

 private:
  Symbol form_;
};

// matched against for special handling of range expressions
struct TORCH_API RangeValue : SugaredValue {
  RangeValue(const SourceRange& loc, Function& m, std::vector<Value*> inputs);
  std::string kind() const override {
    return "range";
  }
  Value* len(const SourceRange& loc, Function& m) override;
  Value* getitem(const SourceRange& loc, Function& m, Value* idx) override;

 private:
  Value* start_;
  Value* end_;
  Value* step_;
  // a flag to determine if it's a simple range() call with only end_ from
  // arguments If true, we will not insert length calculation and index
  // derivation nodes to simplify the graph and enable more possible
  // optimizations
  bool has_only_end_;
};

// Specialized Tree structure to matched against for special handling
// of builtin functions iterables expressions like zip(), enumerate(), etc.
// zip and enumerate can be modeled as a tree of SimpleValue/RangeValue:
//    zip(x, y) ->  (x, y) with tuple assignment to each loop target
//    enumerate(x) -> (range(0, math.inf, 1), x)
// So a complicated expression like zip(a, enumerate(b), range(0, 100)) will be:
// (a, (range(0, math.inf, 1), b), range(0, 100))
// We use those base iterables to fill in the loop information like
// max_trip_count and set the value table for loop targets
struct TORCH_API IterableTree : SugaredValue {
  IterableTree() = default;
  IterableTree(const std::vector<SugaredValuePtr> children)
      : children_(std::move(children)) {}
  std::string kind() const override {
    return "iterabletree";
  }
  void addChild(SugaredValuePtr sv) {
    children_.emplace_back(sv);
  }

  std::vector<SugaredValuePtr> get_children() {
    return children_;
  }

  // given a IterableTree node, get all the base iterables/leaves under the
  // IterableTree node, which are either SimpleValue or RangeValue. This enable
  // us to get all the basic SugaredValues that contains valid loop information
  // with len() and getitem()
  std::vector<SugaredValuePtr> get_base_iterables();

  Value* len(const SourceRange& loc, Function& m) override;
  Value* getitem(const SourceRange& loc, Function& m, Value* idx) override;

 private:
  std::vector<SugaredValuePtr> children_;
};

static inline std::vector<Value*> toValues(
    Graph& g,
    at::ArrayRef<NamedValue> nvs) {
  return fmap(nvs, [&](const NamedValue& v) { return v.value(g); });
}

struct SimpleSelf : public Self {
  explicit SimpleSelf(ClassTypePtr classType)
      : Self(), classType_(std::move(classType)) {}
  std::shared_ptr<SugaredValue> makeSugared(Value* v) const override {
    v->setType(classType_);
    return std::make_shared<SimpleValue>(v);
  }
  ClassTypePtr getClassType() const override {
    return classType_;
  }

 private:
  ClassTypePtr classType_;
};
} // namespace script
} // namespace jit
} // namespace torch
