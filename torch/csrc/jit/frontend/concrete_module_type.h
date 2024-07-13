#pragma once

#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <memory>
#include <string>
#include <vector>

namespace torch {
namespace jit {

enum class IterableModuleKind { NONE, LIST, DICT, PARAMLIST, PARAMDICT };
class ConcreteModuleType;

// You can think of an nn.Module as a template that corresponds to a family of
// JIT types. The template "arguments" are things like the constant values.
// e.g.
//   class M(nn.Module):
//        __constants__ = ["const"]
//        ...
//
// Is similar to writing the following in C++:
//
//    template<TConst>
//    class M {
//       ...
//    }
//
// We need to consider each different member of the type family a different JIT
// type because, e.g. different constant values lead to different versions of
// the same method.
//
// ConcreteModuleType corresponds to a single member of the type family, with
// all template arguments fully specified. Two Modules that share a
// ConcreteModuleType can share a JIT type, and vice versa.
//
// Why not just use a JIT type to represent concrete types? Because constants,
// function attributes, etc. are currently not representable in the type system,
// so this acts a non-first-class way of tracking concrete types.
//
// ConcreteModuleType is also the source of truth for servicing all
// ModuleValue::attr calls. This is so we can guarantee that if two Module's
// share a JIT type (and thus a ConcreteModuleType), then they behave the same
// way when you access attributes on them.

// ConcreteModuleType has two phases.
// 1. Creation: First we build it up, during the ScriptModule conversion
// process. This is represented by ConcreteModuleTypeBuilder.
//    ...then the converter calls ConcreteModuleTypeBuilder::build(), producing
//    a
//       ConcreteModuleType ready for querying.
// 2. Querying: We use ConcreteModuleType as a source of truth for
// ModuleValue::attr calls during method compilation.

// Represents a concrete type during in the process for construction. We use
// this to decide whether we can share types between modules.
class VISIBILITY_HIDDEN ConcreteModuleTypeBuilder {
 public:
  explicit ConcreteModuleTypeBuilder(py::object pyClass) {
    TORCH_INTERNAL_ASSERT(pyClass);
    pyClass_ = std::move(pyClass);
  }

  void addConstant(std::string name, py::object value);
  void addConstant(std::string name, IValue value);
  void addAttribute(
      std::string name,
      const TypePtr& type,
      bool isParameter,
      bool isBuffer);
  void addFunctionAttribute(
      std::string name,
      const TypePtr& type,
      py::object pyFunction);

  void addModule(std::string name, std::shared_ptr<ConcreteModuleType> meta);

  void addForwardHook(py::object hook);
  void addForwardPreHook(py::object pre_hook);

  void addOverload(
      std::string methodName,
      std::vector<std::string> overloadedMethodNames);
  void addBuiltinFunction(std::string name, const std::string& symbol_name);
  void addFailedAttribute(std::string name, std::string failureReason);
  void addIgnoredAttribute(std::string name);
  void setIterableModuleKind(IterableModuleKind kind);

  // If a ConcreteModuleType is poisoned, it will never compare equal to any
  // other concrete type
  void setPoisoned();

  std::shared_ptr<ConcreteModuleType> build() const {
    return std::make_shared<ConcreteModuleType>(*this);
  }

  // This determines whether two modules can share a type. The container structs
  // used by ConcreteModuleType have been defined such that operator==
  // implements a meaningful comparison in that context.
  bool equals(const ConcreteModuleTypeBuilder& other) const;

  struct FunctionAttribute {
    FunctionTypePtr function_;
    py::object pyFunction_;

    friend bool operator==(
        const FunctionAttribute& lhs,
        const FunctionAttribute& rhs) {
      // Functions are not first class, so we can't do type comparison like a
      // regular attribute. So we do a pointer equality check on the actual
      // Python function object.
      return lhs.pyFunction_.is(rhs.pyFunction_);
    }
  };

  struct Attribute {
    Attribute(TypePtr type, bool isParam, bool isBuffer)
        : type_(std::move(type)), isParam_(isParam), isBuffer_(isBuffer) {}

    friend bool operator==(const Attribute& lhs, const Attribute& rhs) {
      return *(lhs.type_) == *(rhs.type_) && lhs.isParam_ == rhs.isParam_;
    }
    TypePtr type_;
    bool isParam_;
    bool isBuffer_;
  };

  struct ModuleInfo {
    ModuleInfo(std::string name, std::shared_ptr<ConcreteModuleType> meta)
        : name_(std::move(name)), meta_(std::move(meta)) {}

    friend bool operator==(const ModuleInfo& lhs, const ModuleInfo& rhs);

    std::string name_;
    std::shared_ptr<ConcreteModuleType> meta_;
  };

 private:
  ConcreteModuleTypeBuilder() = default;
  ClassTypePtr createTypeFromThis() const;

  // If true, this type will never compare equally to anything else. This is
  // used if we want to ensure that this type is not shared (for example, if it
  // came from a traced module)
  bool isPoisoned_ = false;

  // The value of any constants defined by the module.
  std::unordered_map<std::string, IValue> constants_;
  // The types of any attributes
  OrderedDict<std::string, Attribute> attributes_;
  // Overloads, in the same format as `__overloads__` in Python
  std::unordered_map<std::string, std::vector<std::string>> overloads_;
  // Any attributes we failed to convert to TorchScript, along with a hint as to
  // why
  std::unordered_map<std::string, std::string> failedAttributes_;
  // Any attributes that were marked as ignored. They cannot be used in
  // TorchScript but can still be used in ignored function in Python.
  std::unordered_set<std::string> ignoredAttributes_;
  // Any function attributes. These are special right now because functions are
  // not first-class in the type system.
  std::unordered_map<std::string, FunctionAttribute> functionAttributes_;
  // Function attributes that are calls to builtin functions. These get
  // de-sugared directly into the corresponding aten:: call. The map is
  // attribute name -> aten symbol name
  std::unordered_map<std::string, c10::Symbol> builtinFunctions_;
  // The concrete types of any submodules
  std::vector<ModuleInfo> modules_;
  // Hooks to be called before/after forward when the module
  // is called directly. Used to ensure modules have different types
  // when they have different python hooks
  // Actual hooks are added to ClassType directly during compilation
  std::vector<py::object> forwardHooks_;
  std::vector<py::object> forwardPreHooks_;

  // If something is a ModuleDict/ModuleList, it means:
  //   1. The order of the submodules matters for comparing the type
  //   2. The compiler is allowed to treat it like a dict/tuple
  IterableModuleKind iterableModuleKind_ = IterableModuleKind::NONE;

  // The original `nn.Module` class that we derived this ScriptModule from.
  py::object pyClass_;

  // NOTE: If you ever add any more state to this struct, you need to make sure
  // operator== still makes sense!
  friend ConcreteModuleType;
};

// Represents a finalized concrete type, used to service ModuleValue::attr calls
// during method compilation.
class VISIBILITY_HIDDEN ConcreteModuleType {
 public:
  explicit ConcreteModuleType(ConcreteModuleTypeBuilder data);

  static std::shared_ptr<ConcreteModuleType> fromJitType(TypePtr type);

  TypePtr getJitType() const;
  std::optional<py::object> getPyClass() const;
  IterableModuleKind getIterableModuleKind() const;
  std::optional<std::vector<std::string>> findOverloads(
      const std::string& name) const;
  std::optional<Function*> findFunctionAttribute(const std::string& name) const;
  std::optional<c10::Symbol> findBuiltinFunction(const std::string& name) const;
  std::shared_ptr<ConcreteModuleType> findSubmoduleConcreteType(
      const std::string& name) const;
  std::optional<std::string> findFailedAttribute(const std::string& name) const;
  bool isIgnoredAttribute(const std::string& name) const;

  // These getters are only here to return things as types that can be
  // automatically converted by pybind.
  std::unordered_map<std::string, py::object> getConstantsPy() const;
  std::unordered_map<std::string, std::pair<TypePtr, bool>> getAttributesPy()
      const;
  std::vector<std::pair<std::string, std::shared_ptr<ConcreteModuleType>>>
  getModulesPy() const;

  bool equals(const ConcreteModuleType& other) const {
    if (jitType_ == other.jitType_) {
      // If the computed types are the same, these modules can (obviously) share
      // a type.
      return true;
    }

    return data_.equals(other.data_);
  }
  bool equals(const ConcreteModuleTypeBuilder& other) const {
    return data_.equals(other);
  }

  void dump() const;

 private:
  ConcreteModuleType() = default;

  // The JIT type derived from this ConcreteModuleType.
  ConcreteModuleTypeBuilder data_;
  TypePtr jitType_;
};

} // namespace jit
} // namespace torch
