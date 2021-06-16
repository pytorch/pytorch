#pragma once
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include <ATen/core/interned_strings.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/frontend/versioned_symbols.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

using SugaredValuePtr = std::shared_ptr<SugaredValue>;

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

  virtual bool hasAttr(
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

  // TODO @wconstab refactor to use ModuleValue::asTuple instead of new API
  virtual SugaredValuePtr asTupleValue(const SourceRange& loc, Function& m) {
    throw ErrorReport(loc) << kind() << " cannot be used as a tuplevalue";
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
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
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

  // This function is called when to convert a SugaredValue to its iterator.
  // For example, when iterating through a Dict we iterate over its keys
  virtual std::shared_ptr<SugaredValue> iter(
      const SourceRange& loc,
      Function& m) {
    throw ErrorReport(loc) << kind() << " cannot be used as an iterable";
  }

  // If we are iterating over a Sugared Value and it returns a value from this
  // function, then we emit an unrolled loop over the variable. This allows us
  // to support containers of Heterogenous types, like Module Containers &
  // Tuples
  virtual c10::optional<int64_t> staticLen() {
    return c10::nullopt;
  }

  // When iterating over this SugaredValue, should we emit the for loop as an
  // unrolled loop.
  bool shouldEmitUnrolled() {
    return staticLen() != c10::nullopt;
  }

  // return length of this thing, if not then it can't be iterated.
  // If it does not have a statically-determinable length, then it cannot
  // be iterated over with a modulelist. If it does it must return a constant
  // Value *
  virtual Value* len(const SourceRange& loc, Function& m) {
    throw ErrorReport(loc) << "'" << kind() << "'"
                           << " object is not iterable";
  }

  // expression for ith elemement for iterable value
  virtual std::shared_ptr<SugaredValue> getitem(
      const SourceRange& loc,
      Function& m,
      Value* idx,
      TypePtr type_hint = nullptr) {
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
    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    ss << "value of type '" << value_->type()->annotation_str() << "'";
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

  bool hasAttr(const SourceRange& loc, Function& m, const std::string& field)
      override;

  void setAttr(
      const SourceRange& loc,
      Function& m,
      const std::string& field,
      Value* newValue) override;

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      // note: names for args will be 'argument 0', 'argument 1', etc..
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override;

  std::shared_ptr<SugaredValue> iter(const SourceRange& loc, Function& m)
      override;

  Value* getValue() const {
    return value_;
  }

  Value* len(const SourceRange& loc, Function& m) override;
  SugaredValuePtr getitem(
      const SourceRange& loc,
      Function& m,
      Value* idx,
      TypePtr type_hint = nullptr) override;

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
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override;

  // try to create this builtin but if it doesn't exist or the self argument
  // cannot possibly match, then return nullptr. Use in situations where it is
  // not clear if it is a valid builtin
  static std::shared_ptr<BuiltinFunction> tryCreate(
      Symbol symbol,
      c10::optional<NamedValue> self);
};

struct TORCH_API SugaredTupleValue : public SugaredValue {
  explicit SugaredTupleValue(std::vector<std::shared_ptr<SugaredValue>> tup)
      : tup_(std::move(tup)){};

  std::vector<std::shared_ptr<SugaredValue>> asTuple(
      const SourceRange& loc,
      Function& m,
      const c10::optional<size_t>& size_hint = {}) override {
    return tup_;
  };

  Value* asValue(const SourceRange& loc, Function& m) override {
    std::vector<Value*> vec;
    for (const auto& sv : tup_) {
      vec.push_back(sv->asValue(loc, m));
    }
    Graph& g = *m.graph();
    return g.insertNode(g.createTuple(vec))->output();
  }

  std::string kind() const override {
    return "Tuple";
  }

  SugaredValuePtr getitem(
      const SourceRange& loc,
      Function& m,
      Value* idx,
      TypePtr type_hint = nullptr) override {
    if (!(idx->type()->cast<IntType>() && toIValue(idx))) {
      throw ErrorReport(loc)
          << "Expected integer literal for index. "
          << "ModuleList/Sequential indexing is only supported with integer literals. "
          << "Enumeration is supported, e.g. 'for index, v in enumerate(self): ...'";
    }
    auto index = toIValue(idx)->toInt();
    int64_t adj_index =
        (index < 0) ? index + static_cast<int64_t>(tup_.size()) : index;
    if (!(adj_index >= 0 && adj_index < static_cast<int64_t>(tup_.size()))) {
      throw ErrorReport(loc)
          << "Index " << index << " out of range of length " << tup_.size();
    }
    return tup_.at(adj_index);
  }

  // This function is called when a SugaredValue is used to convert a
  // SugaredValue to its iterator. For example, when iterating through a Dict we
  // iterate over its keys
  std::shared_ptr<SugaredValue> iter(const SourceRange& loc, Function& m)
      override {
    return shared_from_this();
  };

  // Because this is used to contain SugaredValues of Heterogenous types,
  // we define staticLen() so that when this is iterated over it is emitted
  // as an unrolled loop.
  c10::optional<int64_t> staticLen() override {
    return static_cast<int64_t>(tup_.size());
  }

  std::vector<std::shared_ptr<SugaredValue>> tup_;
};

struct TORCH_API BuiltinModule : public SugaredValue {
  BuiltinModule(std::string name, c10::optional<int64_t> version = at::nullopt)
      : name(std::move(name)), version(version) {}

  std::string kind() const override {
    return "builtin module";
  }
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) override {
    if (field == "autograd") {
      // When refering torch.autograd, it is also considered to be a
      // BuiltinModule and we will dispatch to the aten operators for the
      // methods under its module.
      return std::make_shared<BuiltinModule>("aten", version);
    }

    auto sym = Symbol::fromQualString(name + "::" + field);
    if (version.has_value()) {
      // Possibly replaces symbol with another that implements its
      // historic behavior.
      // See note [Versioned Symbols]
      sym = get_symbol_for_version(sym, *version);
    }
    return std::make_shared<BuiltinFunction>(sym, c10::nullopt);
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
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
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
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override;

  std::string kind() const override {
    return type_->str();
  }

  TupleTypePtr type_;
};

struct FunctionValue : public SugaredValue {
  FunctionValue(Function* callee) : callees_({callee}) {}
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
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override {
    std::vector<const FunctionSchema*> schemas;
    for (Function* callee : callees_) {
      try {
        callee->ensure_defined();
      } catch (const RecursiveMethodCallError&) {
        throw ErrorReport(loc)
            << " function '" << callee->name() << "' is called recursively. "
            << "Recursive calls are not supported";
      }
      schemas.push_back(&callee->getSchema());
    }
    auto match = matchSchemas(schemas, loc, *f.graph(), args, kwargs);
    Value* output =
        f.graph()->insertFunctionCall(callees_[match.first], match.second);
    output->node()->setSourceRange(loc);
    return std::make_shared<SimpleValue>(output);
  }

  const std::vector<Function*>& callees() {
    return callees_;
  }

 private:
  std::vector<Function*> callees_;
  // TODO holding this thing is creepy
  std::shared_ptr<CompilationUnit> cu_;
};

struct TORCH_API ClosureValue : public SugaredValue {
  ClosureValue(Value* value) : value_(value) {
    TORCH_INTERNAL_ASSERT(value_->node()->kind() == prim::Closure);
  }
  std::string kind() const override {
    return "closure";
  }
  Value* asValue(const SourceRange& range, Function& m) override {
    return value_;
  }
  Value* value_;
};

// defines how a method obtained from a module/class/interface behaves in script
struct MethodValue : public SugaredValue {
  MethodValue(Value* self, std::vector<std::string> method_names)
      : self_(self), method_names_(std::move(method_names)) {}
  MethodValue(Value* self, std::string method_name)
      : MethodValue(self, std::vector<std::string>({std::move(method_name)})) {}

  std::string kind() const override {
    return "method";
  }

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& f,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override {
    std::vector<NamedValue> argsWithSelf = {self_};
    argsWithSelf.insert(argsWithSelf.end(), args.begin(), args.end());
    std::vector<const FunctionSchema*> schemas;
    for (const std::string& method_name : method_names_) {
      if (auto class_type = self_->type()->cast<ClassType>()) {
        Function& method = class_type->getMethod(method_name);
        try {
          method.ensure_defined();
        } catch (const RecursiveMethodCallError&) {
          throw ErrorReport(loc)
              << " method '" << method.name() << "' is called recursively. "
              << "Recursive calls are not supported";
        }
        schemas.push_back(&method.getSchema());
      } else if (auto interface_type = self_->type()->cast<InterfaceType>()) {
        schemas.push_back(interface_type->getMethod(method_name));
      } else {
        TORCH_INTERNAL_ASSERT(
            false, "method constructed that is not a class or interface");
      }
    }
    auto match = matchSchemas(schemas, loc, *f.graph(), argsWithSelf, kwargs);
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
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
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
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override {
    if (args.size() == 1 && kwargs.size() == 0) {
      auto len_op = std::make_shared<BuiltinFunction>(aten::len, at::nullopt);
      auto gt_op = std::make_shared<BuiltinFunction>(aten::gt, at::nullopt);
      auto zero = m.graph()->insertConstant(0);

      auto v = args[0].value(*m.graph());
      if (v->type()->isSubtypeOf(type_)) {
        return std::make_shared<SimpleValue>(v);
      } else if (
          *type_ == *BoolType::get() &&
          (v->type()->isSubtypeOf(AnyListType::get()) ||
           v->type()->isSubtypeOf(StringType::get()) ||
           v->type()->cast<DictType>())) {
        auto len = len_op->call(loc, m, {v}, {}, 1);
        return gt_op->call(loc, m, {len->asValue(loc, m), zero}, {}, 1);
      }
    }
    return BuiltinFunction::call(loc, m, args, kwargs, n_binders);
  }

 private:
  TypePtr type_;
};

struct TORCH_API TensorCastValue : public SugaredValue {
  TensorCastValue(at::ScalarType type, NamedValue self)
      : dtype_(type), self_(std::move(self)) {}

  std::string kind() const override {
    return "Cast";
  }

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override {
    TORCH_INTERNAL_ASSERT(args.size() == 0 && kwargs.size() == 0);
    Value* dtype_const = m.graph()->insertConstant(dtype_, loc);
    std::vector<NamedValue> kwargs_{
        self_, NamedValue(loc, "dtype", dtype_const)};
    Value* casted_val = m.graph()->insert(
        /*opname=*/Symbol::fromQualString("aten::to"),
        /*args=*/args,
        /*kwargs=*/kwargs_,
        /*range=*/loc);
    return std::make_shared<SimpleValue>(casted_val);
  }

  at::ScalarType dtype_;
  NamedValue self_;
};

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
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
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
  RangeValue(
      const SourceRange& loc,
      Function& m,
      std::vector<Value*> input,
      c10::optional<int64_t> static_len = c10::nullopt);

  std::string kind() const override {
    return "range";
  }
  Value* len(const SourceRange& loc, Function& m) override;
  SugaredValuePtr getitem(
      const SourceRange& loc,
      Function& m,
      Value* idx,
      TypePtr type_hint = nullptr) override;
  std::shared_ptr<SugaredValue> iter(const SourceRange& loc, Function& m)
      override;

  // When Range is instantiated via enumerate(iterable_with_static_len),
  // then it takes the static length of the iterable
  c10::optional<int64_t> staticLen() override {
    return static_len_;
  }

 private:
  Value* start_;
  Value* end_;
  Value* step_;
  // a flag to determine if it's a simple range() call with only end_ from
  // arguments If true, we will not insert length calculation and index
  // derivation nodes to simplify the graph and enable more possible
  // optimizations
  bool has_only_end_;
  c10::optional<int64_t> static_len_ = c10::nullopt;
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
// Iterables can contain lists of SugaredValues like ModuleLists. If it
// does, then we emit it unrolled and require that all values it contains
// have a statically-determinable length.
struct TORCH_API IterableTree : SugaredValue {
  IterableTree() = default;
  IterableTree(
      const SourceRange& range,
      Function& m,
      at::ArrayRef<SugaredValuePtr> children) {
    for (const auto& child : children) {
      addChild(range, m, child);
    }
  }
  std::string kind() const override {
    return "iterabletree";
  }

  std::shared_ptr<SugaredValue> iter(const SourceRange& loc, Function& m)
      override {
    return shared_from_this();
  }

  void addChild(
      const SourceRange& range,
      Function& m,
      const SugaredValuePtr& iter_value);

  std::vector<SugaredValuePtr> get_children() {
    return children_;
  }

  // If this iterable contains a ModuleList or Tuple, then it will have a
  // static length, and we will emit it as an unrolled for loop.
  c10::optional<int64_t> staticLen() override {
    return unroll_length_;
  }

  // given a IterableTree node, get all the base iterables/leaves under the
  // IterableTree node. This enables
  // us to get all the basic SugaredValues that contains valid loop information
  // with len() and getitem()
  std::vector<SugaredValuePtr> get_base_iterables();

  Value* len(const SourceRange& loc, Function& m) override;
  SugaredValuePtr getitem(
      const SourceRange& loc,
      Function& m,
      Value* idx,
      TypePtr type_hint = nullptr) override;

 private:
  c10::optional<int64_t> unroll_length_ = c10::nullopt;
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

// This is not a SimpleValue so it can not pass through the code paths that
// expect a SimpleValue as a sugared value.
struct TORCH_API ExceptionMessageValue : public SugaredValue {
  explicit ExceptionMessageValue(Value* value) : value_(value) {}

  std::string kind() const override {
    return "exception message";
  }

  Value* getValue() {
    return value_;
  }

  Value* value_;
};

struct TORCH_API ExceptionValue : public SugaredValue {
  explicit ExceptionValue(std::string message) : message_(std::move(message)) {}

  std::string kind() const override {
    return "exception";
  }

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& m,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> /*attributes*/,
      size_t /*n_binders*/) override {
    auto exception_message = insertConstant(*m.graph(), message_ + ": ", loc);
    for (auto& input : args) {
      auto input_str = input.value(*m.graph());
      if (!input_str->type()->isSubtypeOf(StringType::get())) {
        input_str =
            emitBuiltinCall(loc, *m.graph(), aten::str, {input_str}, {});
      }
      exception_message = emitBuiltinCall(
          loc, *m.graph(), aten::add, {exception_message, input_str}, {});
    }
    return std::make_shared<ExceptionMessageValue>(exception_message);
  }

  std::string message_;
};

struct TORCH_API SugaredEnumClass : public SugaredValue {
  explicit SugaredEnumClass(EnumTypePtr enum_type)
      : enum_type_(std::move(enum_type)) {}

  std::string kind() const override {
    return "EnumClass";
  }

  SugaredValuePtr attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) override;

  SugaredValuePtr iter(const SourceRange& loc, Function& m) override;

 private:
  EnumTypePtr enum_type_;
};

struct TORCH_API SliceValue : public SugaredValue {
  explicit SliceValue(Value* start, Value* stop, Value* step)
      : start_(start), stop_(stop), step_(step) {}

  std::string kind() const override {
    return "Python slice value";
  }

  Value* start() {
    return start_;
  };
  Value* stop() {
    return stop_;
  };
  Value* step() {
    return step_;
  };

 private:
  Value* start_;
  Value* stop_;
  Value* step_;
};

} // namespace jit
} // namespace torch
