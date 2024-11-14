#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/concrete_module_type.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace torch::jit {

std::string typeString(py::handle h);

inline std::shared_ptr<SugaredValue> toSimple(Value* v) {
  return std::make_shared<SimpleValue>(v);
}

// NB: This should be the single entry-point for instantiating a SugaredValue
// from a Python object. If you are adding support for converting a new Python
// type, *add it in this function's implementation*.
std::shared_ptr<SugaredValue> toSugaredValue(
    py::object obj,
    GraphFunction& m,
    const SourceRange& loc,
    bool is_constant = false);

std::optional<StrongFunctionPtr> as_function(const py::object& obj);

struct VISIBILITY_HIDDEN PythonValue : public SugaredValue {
  PythonValue(
      py::object the_self,
      std::optional<py::object> rcb = std::nullopt,
      Value* module_self = nullptr)
      : self(std::move(the_self)),
        rcb(std::move(rcb)),
        moduleSelf_(module_self) {}

  FunctionSchema getSchema(
      const size_t n_args,
      const size_t n_binders,
      const SourceRange& loc);

  // call it like a function, e.g. `outputs = this(inputs)`
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& m,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override;

  std::string kind() const override;

  std::vector<std::shared_ptr<SugaredValue>> asTuple(
      const SourceRange& loc,
      GraphFunction& m,
      const std::optional<size_t>& size_hint = {}) override;

  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override;

  Value* asValue(const SourceRange& loc, GraphFunction& m) override {
    throw(
        ErrorReport(loc)
        << kind() << " cannot be used as a value. "
        << "Perhaps it is a closed over global variable? If so, please "
        << "consider passing it in as an argument or use a local varible "
        << "instead.");
  }

 protected:
  py::object getattr(const SourceRange& loc, const std::string& name);

  void checkForAddToConstantsError(std::stringstream& ss);

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  py::object self;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::optional<py::object> rcb;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  Value* moduleSelf_ = nullptr;
};

struct VISIBILITY_HIDDEN PythonModuleValue : public PythonValue {
  explicit PythonModuleValue(py::object mod) : PythonValue(std::move(mod)) {}

  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override;
};

// Used for desugaring uses of the torch.cuda module. All the CUDA APIs with
// torch.cuda.* are resolved using CUDAPythonModuleValue.
struct VISIBILITY_HIDDEN CUDAPythonModuleValue : public PythonValue {
  explicit CUDAPythonModuleValue(py::object mod)
      : PythonValue(std::move(mod)) {}

  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override;
};

// Represents all the parameters of a module as a List[Tensor]
struct VISIBILITY_HIDDEN ConstantParameterList : public SugaredValue {
  ConstantParameterList(Value* the_list) : the_list_(the_list) {}
  std::string kind() const override {
    return "constant parameter list";
  }
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& caller,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override {
    return toSimple(the_list_);
  }

 private:
  Value* the_list_;
};

struct VISIBILITY_HIDDEN ModuleDictMethod : public SugaredValue {
  explicit ModuleDictMethod(SugaredValuePtr iterable, std::string name)
      : iterable_(std::move(iterable)), name_(std::move(name)) {}

  std::string kind() const override {
    return name_;
  }

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& f,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override {
    if (!args.empty() || !kwargs.empty()) {
      throw(
          ErrorReport(loc) << name_ << " method does not accept any arguments");
    }
    return iterable_;
  }

  SugaredValuePtr iterable_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::string name_;
};

struct SugaredDict;

// defines how modules/methods behave inside the script subset.
// for now this does not have any interaction with python.
// in the future, we will add the ability to resolve `self.foo` to python
// {functions, modules, constants} so this SugaredValue is defined here
// anticipating we will eventually need to replace Module with a py::object
// holding the actual nn.Module class.

struct VISIBILITY_HIDDEN ModuleValue : public SugaredValue {
  ModuleValue(Value* self, std::shared_ptr<ConcreteModuleType> concreteType)
      : self_(self), concreteType_(std::move(concreteType)) {}

  std::string kind() const override {
    return "module";
  }

  Value* asValue(const SourceRange& loc, GraphFunction& m) override;

  SugaredValuePtr asTupleValue(const SourceRange& loc, GraphFunction& m)
      override;

  // select an attribute on it, e.g. `this.field`
  std::shared_ptr<SugaredValue> tryGetAttr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field);

  // select an attribute on it, e.g. `this.field`
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override;

  // select an attribute on it, e.g. `this.field`
  bool hasAttr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override;

  // call module.forward with pre_hooks and hooks
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& caller,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override;

  std::shared_ptr<SugaredDict> getSugaredDict(
      const SourceRange& loc,
      GraphFunction& m);

  std::shared_ptr<SugaredDict> getSugaredNamedBufferDict(
      const SourceRange& loc,
      GraphFunction& m);

  std::shared_ptr<SugaredDict> getSugaredNamedParameterList(
      const SourceRange& loc,
      GraphFunction& m);

  std::shared_ptr<SugaredDict> getSugaredNamedParameterDict(
      const SourceRange& loc,
      GraphFunction& m);

  void setAttr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field,
      Value* newValue) override;

  SugaredValuePtr iter(const SourceRange& loc, GraphFunction& m) override;

  std::shared_ptr<SugaredValue> getitem(
      const SourceRange& loc,
      GraphFunction& m,
      Value* idx,
      TypePtr type_hint) override;

 private:
  // Check that the type of all submodules is a subtype of ty. If the function
  // returns false, more information about why it returns false (e.g. which
  // submodule's type is not a subtype of ty) is printed it why_not if it is not
  // null.
  bool areAllSubmodulesSubtypeOf(
      const TypePtr& ty,
      std::ostream* why_not = nullptr) const;

  Value* self_;
  std::shared_ptr<ConcreteModuleType> concreteType_;
};

bool isNamedTupleClass(const py::object& obj);
TypePtr registerNamedTuple(
    const py::object& obj,
    const SourceRange& loc,
    const ResolutionCallback& rcb);

void recurseThroughNestedModules(
    const SourceRange& loc,
    GraphFunction& m,
    std::vector<SugaredValuePtr>& keys,
    std::vector<SugaredValuePtr>& values,
    std::shared_ptr<ModuleValue>& self,
    const std::string& prefix,
    const std::string& field);

// Used to support named_modules()
struct VISIBILITY_HIDDEN SugaredDict : public SugaredValue {
  explicit SugaredDict(
      std::shared_ptr<ModuleValue> self,
      std::shared_ptr<SugaredTupleValue> keys,
      std::shared_ptr<SugaredTupleValue> modules)
      : self_(std::move(self)),
        keys_(std::move(keys)),
        modules_(std::move(modules)) {}

  std::string kind() const override {
    return "ModuleDict";
  }

  std::shared_ptr<SugaredTupleValue> getKeys() {
    return keys_;
  }

  std::shared_ptr<SugaredTupleValue> getModules() {
    return modules_;
  }

  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override;

  SugaredValuePtr iter(const SourceRange& loc, GraphFunction& m) override {
    return keys_;
  }

  std::shared_ptr<ModuleValue> self_;
  std::shared_ptr<SugaredTupleValue> keys_;
  std::shared_ptr<SugaredTupleValue> modules_;
};

struct VISIBILITY_HIDDEN BooleanDispatchValue : public SugaredValue {
  BooleanDispatchValue(py::dict dispatched_fn)
      : dispatched_fn_(std::move(dispatched_fn)) {}

  std::string kind() const override {
    return "boolean dispatch";
  }

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& caller,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override;

 private:
  py::dict dispatched_fn_;
};

struct VISIBILITY_HIDDEN PythonClassValue : public ClassValue {
  PythonClassValue(ClassTypePtr type, py::object py_type)
      : ClassValue(std::move(type)), py_type_(std::move(py_type)) {}

  std::string kind() const override {
    return "Python type";
  }

  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override;

  bool hasAttr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override;

 private:
  py::object py_type_;
};

struct VISIBILITY_HIDDEN PythonExceptionValue : public ExceptionValue {
  explicit PythonExceptionValue(const py::object& exception_class)
      : ExceptionValue(
            py::str(py::getattr(exception_class, "__name__", py::str("")))),
        exception_class_qualified_name_(
            py::str(py::module::import("torch._jit_internal")
                        .attr("_qualified_name")(
                            exception_class,
                            /*mangle_name=*/false))) {}

  std::string kind() const override {
    return "Python exception";
  }

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& caller,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override;

 private:
  std::string exception_class_qualified_name_;
};

// Python Slice class.
struct VISIBILITY_HIDDEN PythonSliceClass : public SugaredValue {
  explicit PythonSliceClass() = default;

  std::string kind() const override {
    return "Python slice class";
  }

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      GraphFunction& caller,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs,
      size_t n_binders) override;
};

} // namespace torch::jit
