#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/concrete_module_type.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace torch {
namespace jit {

std::string typeString(py::handle h);

inline std::shared_ptr<SugaredValue> toSimple(Value* v) {
  return std::make_shared<SimpleValue>(v);
}

// NB: This should be the single entry-point for instantiating a SugaredValue
// from a Python object. If you are adding support for converting a new Python
// type, *add it in this function's implementation*.
std::shared_ptr<SugaredValue> toSugaredValue(
    py::object obj,
    Function& m,
    SourceRange loc,
    bool is_constant = false);

c10::optional<StrongFunctionPtr> as_function(const py::object& obj);

struct VISIBILITY_HIDDEN PythonValue : public SugaredValue {
  PythonValue(
      py::object the_self,
      c10::optional<py::object> rcb = c10::nullopt,
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
      Function& m,
      at::ArrayRef<NamedValue> inputs_,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override;

  std::string kind() const override;

  std::vector<std::shared_ptr<SugaredValue>> asTuple(
      const SourceRange& loc,
      Function& m,
      const c10::optional<size_t>& size_hint = {}) override;

  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) override;

 protected:
  py::object getattr(const SourceRange& loc, const std::string& name);

  void checkForAddToConstantsError(std::stringstream& ss);

  py::object self;
  c10::optional<py::object> rcb;
  Value* moduleSelf_ = nullptr;
};

struct VISIBILITY_HIDDEN PythonModuleValue : public PythonValue {
  explicit PythonModuleValue(py::object mod) : PythonValue(std::move(mod)) {}

  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
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
      Function& caller,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override {
    return toSimple(the_list_);
  }

 private:
  Value* the_list_;
};


struct VISIBILITY_HIDDEN ModuleDictMethod : public SugaredValue {
  explicit ModuleDictMethod(SugaredValuePtr iterable, const std::string& name)
      : iterable_(iterable), name_(name){};

  std::string kind() const override {
    return name_;
  }

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& f,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override {
    if (inputs.size() || attributes.size()) {
      throw ErrorReport(loc)
          << name_ << " method does not accept any arguments";
    }
    return iterable_;
  }

  SugaredValuePtr iterable_;
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

  Value* asValue(const SourceRange& loc, Function& m) override;

  // select an attribute on it, e.g. `this.field`
  std::shared_ptr<SugaredValue> tryGetAttr(
      const SourceRange& loc,
      Function& m,
      const std::string& field);

  // select an attribute on it, e.g. `this.field`
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) override;

  // select an attribute on it, e.g. `this.field`
  bool hasAttr(const SourceRange& loc, Function& m, const std::string& field)
      override;

  // call module.forward
  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& caller,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override {
    return attr(loc, caller, "forward")
        ->call(loc, caller, inputs, attributes, n_binders);
  }

  std::shared_ptr<SugaredDict> getSugaredDict(
      const SourceRange& loc,
      Function& m);

  std::shared_ptr<SugaredDict> getSugaredNamedBufferDict(
      const SourceRange& loc,
      Function& m);

  std::shared_ptr<SugaredDict> getSugaredNamedParametersDict(
      const SourceRange& loc,
      Function& m);

  void setAttr(
      const SourceRange& loc,
      Function& m,
      const std::string& field,
      Value* newValue) override;

  SugaredValuePtr iter(const SourceRange& loc, Function& m) override;

  std::shared_ptr<SugaredValue> getitem(
      const SourceRange& loc,
      Function& m,
      Value* idx) override;

  std::shared_ptr<ConcreteModuleType> getConcreteType() {
    return concreteType_;
  }

  Value* getSelf() {
    return self_;
  }

 private:
  Value* self_;
  std::shared_ptr<ConcreteModuleType> concreteType_;
};

bool isNamedTupleClass(const py::object& obj);
TypePtr registerNamedTuple(const py::object& obj, const SourceRange& loc);

void recurseThroughNestedModules(
    const SourceRange& loc,
    Function& m,
    std::vector<SugaredValuePtr>& keys,
    std::vector<SugaredValuePtr>& values,
    std::shared_ptr<ModuleValue> self,
    const std::string& prefix,
    const std::string& field,
    std::function<void(std::shared_ptr<ModuleValue>)> const& onModuleCallback = {});

// Used to support named_modules()
struct VISIBILITY_HIDDEN SugaredDict : public SugaredValue {
  explicit SugaredDict(
      std::shared_ptr<ModuleValue> self,
      std::shared_ptr<SugaredTupleValue> keys,
      std::shared_ptr<SugaredTupleValue> modules) {
    self_ = std::move(self);
    keys_ = std::move(keys);
    modules_ = std::move(modules);
  }

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
      Function& m,
      const std::string& field) override;

  SugaredValuePtr iter(const SourceRange& loc, Function& m) {
    return keys_;
  };

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
      Function& caller,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
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
      Function& m,
      const std::string& field) override;

  bool hasAttr(const SourceRange& loc, Function& m, const std::string& field)
      override;

 private:
  py::object py_type_;
};

struct VISIBILITY_HIDDEN ModuleDictMethodRecursive : public SugaredValue {
  explicit ModuleDictMethodRecursive(std::shared_ptr<ModuleValue> module,
  std::shared_ptr<SugaredTupleValue> keys,
  std::shared_ptr<SugaredTupleValue> values, 
  const std::string& name)
      : module_(module), keys_(keys), values_(values), name_(name){};

  std::string kind() const override {
    return name_;
  }

  std::shared_ptr<SugaredValue> call(
      const SourceRange& loc,
      Function& f,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes,
      size_t n_binders) override {
      auto iterator = std::make_shared<IterableTree>();

      std::map<std::string, NamedValue> inputMap;
      for (const auto& input : inputs) {
        if (input.hasName()) {
          inputMap.insert({input.name(), input});
        } else {
          throw ErrorReport(loc)
              << "Please explicitly name all parameters passed to: " << name_ << "()";
        }
      }

      if (inputMap.find("prefix") != inputMap.end()) {
          throw ErrorReport(loc)
            << "Prefix is not currently supported yet for: " << name_ << "()";
      }
      
      if (inputMap.find("recurse") != inputMap.end()) {
        auto recurseInputValue = inputMap.find("recurse")->second.value(*f.graph());
        if (recurseInputValue->type()->kind() == TypeKind::BoolType) {
          bool shouldRecurse = constant_as<bool>(recurseInputValue).value();
          if (shouldRecurse) {
            std::vector<SugaredValuePtr> moduleKeys;
            std::vector<SugaredValuePtr> moduleValues;

             std::vector<SugaredValuePtr> iterValues;
             std::vector<SugaredValuePtr> iterKeys;  
            
              auto lambda = [&](std::shared_ptr<ModuleValue> m) -> void { 
                std::vector<std::string> names;
                const auto& selfType = m->getConcreteType()->getJitType()->expect<ClassType>();
                for (size_t i = 0; i < selfType->numAttributes(); ++i) {
                  if (name_ == "named_parameters") {
                    if (selfType->is_parameter(i)) {
                      names.push_back(selfType->getAttributeName(i));
                    }
                  } else if (name_ == "named_buffers") { 
                    if (selfType->is_buffer(i)) {
                      names.push_back(selfType->getAttributeName(i));
                    }
                  } else {
                    throw ErrorReport(loc)
                    << " Recursive iteration attempted on an unsupported member " << name_;
                  }
                } 

                for (const auto& item_name : names) {
                  auto name_v =
                      std::make_shared<SimpleValue>(insertConstant(*f.graph(), item_name));
                  Value* tensor_v = f.graph()->insertGetAttr(m->getSelf(), item_name);
                  iterValues.push_back(m->tryGetAttr(loc, f, item_name));
                  iterKeys.push_back(name_v);
                } 
             }; 
            recurseThroughNestedModules(loc, f, moduleKeys, moduleValues, module_, "", name_, lambda);
            auto iterator = std::make_shared<IterableTree>();
            iterator->addChild(loc, f, std::make_shared<SugaredTupleValue>(iterKeys));
            iterator->addChild(loc, f, std::make_shared<SugaredTupleValue>(iterValues));
            return iterator;
          }
        } else {
          throw ErrorReport(loc)
            << name_ << " method expects no argument, or boolean argument.";
        }
      }
      // No args, means not recursive
      if (name_ == "named_parameters") {
        auto iterator = std::make_shared<IterableTree>();
        auto kvNamedBufferDict =  module_->getSugaredNamedParametersDict(loc, f);
        iterator->addChild(loc, f, kvNamedBufferDict->getKeys());
        iterator->addChild(loc, f, kvNamedBufferDict->getModules());
        return iterator;
      } else if (name_ == "named_buffers") {
        auto iterator = std::make_shared<IterableTree>();
        auto kvNamedBufferDict =  module_->getSugaredNamedBufferDict(loc, f);
        iterator->addChild(loc, f, kvNamedBufferDict->getKeys());
        iterator->addChild(loc, f, kvNamedBufferDict->getModules());
        return iterator;
      } else {
        throw ErrorReport(loc)
            << "Iteration attempted on an unsupported member " << name_;
      }
      
    }

  std::shared_ptr<ModuleValue> module_;
  std::shared_ptr<SugaredTupleValue> keys_;
  std::shared_ptr<SugaredTupleValue> values_;
  const std::string name_;
};

} // namespace jit
} // namespace torch
