#pragma once

#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/jit/script/module.h>
#include <memory>
#include <string>
#include <vector>

namespace torch {
namespace jit {
namespace script {
enum class IterableModuleKind { NONE, LIST, DICT };

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
class VISIBILITY_HIDDEN ConcreteModuleType {
 public:
  // ConcreteModuleType has two states.
  // 1. Building: First we build it up, during the ScriptModule conversion
  // process
  //      ... to transition, we freeze the type by associating with a JIT type.
  // 2. Querying: Then we ask use it as a source of truth during method
  // compilation.
  //              During this time the ModuleType is effectively const.
  // Yes, it could be two different types. Not terribly worth the verbosity.

  /**
   * Builder methods (jitType must be null)
   */
  void addPyClass(py::object pyClass);
  void addConstant(std::string name, py::object value);
  void addAttribute(std::string name, TypePtr type, bool isParameter);
  void addFunctionAttribute(
      std::string name,
      const TypePtr& type,
      py::object pyFunction);

  // add a submodule to the ConcreteModuleType can either be ConcreteModuleType
  // that get constructed recursively, or InterfaceType (Module)
  void addModule(std::string name, std::shared_ptr<ConcreteModuleType> meta);
  void addModuleInterface(std::string name, const TypePtr& type);

  void addOverload(
      std::string methodName,
      std::vector<std::string> overloadedMethodNames);
  void addFailedAttribute(std::string name, std::string failureReason);
  void setIterableModuleKind(IterableModuleKind kind);
  void setPoisoned();

  /**
   * Freezing methods
   */

  // Based on the data in this ConcreteType, create an equivalent JIT type and
  // associate this module type with it.
  ClassTypePtr createNewTypeFromThis();
  // Associate the provided type with this ConcreteType
  void addJitType(ClassTypePtr type);

  /**
   * Query methods (jitType must be non-null)
   */
  ClassTypePtr getJitType() const;
  py::object getPyClass() const;
  IterableModuleKind getIterableModuleKind() const;
  c10::optional<py::object> findConstant(const std::string& name) const;
  c10::optional<std::vector<std::string>> findOverloads(
      const std::string& name) const;
  c10::optional<Function*> findFunctionAttribute(const std::string& name) const;
  std::shared_ptr<ConcreteModuleType> findSubmoduleConcreteType(
      const std::string& name) const;
  c10::optional<std::string> findFailedAttribute(const std::string& name) const;

  // These getters are only here to return things as types that can be
  // automatically converted by pybind.
  std::unordered_map<std::string, py::object> getConstantsPy() const;
  std::unordered_map<std::string, std::pair<TypePtr, bool>> getAttributesPy()
      const;
  std::vector<std::pair<std::string, TypePtr>> getModulesPy() const;

  // This determines whether two modules can share a type. The container structs
  // used by ConcreteModuleType have been defined such that operator==
  // implements a meaningful comparison in that context.
  friend bool operator==(
      const ConcreteModuleType& lhs,
      const ConcreteModuleType& rhs) {
    if (lhs.jitType_ == rhs.jitType_) {
      // If the computed types are the same, these modules can (obviously) share
      // a type.
      return true;
    }

    if (lhs.isPoisoned_ || rhs.isPoisoned_) {
      return false;
    }

    // clang-format off
    // These are vaguely ordered so that cheap, discriminating checks happen first.
    bool equal =
      lhs.pyClass_.is(rhs.pyClass_) &&
      lhs.iterableModuleKind_ == rhs.iterableModuleKind_ &&
      lhs.constants_ == rhs.constants_ &&
      lhs.attributes_ == rhs.attributes_ &&
      lhs.overloads_ == rhs.overloads_ &&
      lhs.functionAttributes_ == rhs.functionAttributes_;
    // clang-format on
    if (!equal) {
      return false;
    }

    // We store modules in order of insertion (to make compilation
    // deterministic). However, for the purposes of equality, insertion order
    // should not matter, so sort them by name.
    // We put this check last because it involves the most work.
    auto lhsSorted = lhs.modules_;
    std::sort(
        lhsSorted.begin(),
        lhsSorted.end(),
        [](const ModuleInfo& a, const ModuleInfo& b) {
          return a.name_ < b.name_;
        });

    auto rhsSorted = rhs.modules_;
    std::sort(
        rhsSorted.begin(),
        rhsSorted.end(),
        [](const ModuleInfo& a, const ModuleInfo& b) {
          return a.name_ < b.name_;
        });

    return lhsSorted == rhsSorted;
  }

  void dump() const;

 private:
  struct Constant {
    /* implicit */ Constant(py::object v) : v_(std::move(v)) {}
    friend bool operator==(const Constant& lhs, const Constant& rhs) {
      // Perform the equivalent of `lhs == rhs` in Python.
      int rv = PyObject_RichCompareBool(lhs.v_.ptr(), rhs.v_.ptr(), Py_EQ);
      if (rv == -1) {
        throw py::error_already_set();
      }
      return rv == 1;
    }
    py::object v_;
  };

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
    Attribute(TypePtr type, bool isParam)
        : type_(std::move(type)), isParam_(isParam) {}

    friend bool operator==(const Attribute& lhs, const Attribute& rhs) {
      return *(lhs.type_) == *(rhs.type_) && lhs.isParam_ == rhs.isParam_;
    }
    TypePtr type_;
    bool isParam_;
  };

  struct ModuleInfo {
    ModuleInfo(std::string name, std::shared_ptr<ConcreteModuleType> meta)
        : name_(std::move(name)), meta_(std::move(meta)), type_(nullptr) {}

    ModuleInfo(std::string name, const TypePtr& type)
        : name_(std::move(name)), meta_(nullptr), type_(type) {}
    friend bool operator==(const ModuleInfo& lhs, const ModuleInfo& rhs) {
      if (lhs.meta_ != nullptr && rhs.meta_ != nullptr) {
        return *(lhs.meta_) == *(rhs.meta_);
      } else if (lhs.type_ != nullptr && rhs.type_ != nullptr) {
        return  *(lhs.type_) == *(rhs.type_);
      } else {
        return false;
      }
    }

    TypePtr getJitType() const {
      return meta_ == nullptr? type_ : meta_->getJitType();
    }
    std::string name_;

    // Module Info contains either an ConcreateModuleType or a type (which is
    // a Module Interface), these two are union relationship.
    std::shared_ptr<ConcreteModuleType> meta_;
    TypePtr type_;

  };

  // If true, this type will never compare equally to anything else. This is
  // used if we want to ensure that this type is not shared (for example, if it
  // came from a traced module)
  bool isPoisoned_ = false;

  // The value of any constants defined by the module.
  std::unordered_map<std::string, Constant> constants_;
  // The types of any attributes
  std::unordered_map<std::string, Attribute> attributes_;
  // Overloads, in the same format as `__overloads__` in Python
  std::unordered_map<std::string, std::vector<std::string>> overloads_;
  // Any attributes we failed to convert to TorchScript, along with a hint as to
  // why
  std::unordered_map<std::string, std::string> failedAttributes_;
  // Any function attributes. These are special right now because functions are
  // not first-class in the type system.
  std::unordered_map<std::string, FunctionAttribute> functionAttributes_;
  // The concrete types of any submodules
  std::vector<ModuleInfo> modules_;

  // If something is a ModuleDict/ModuleList, it means:
  //   1. The order of the submodules matters for comparing the type
  //   2. The compiler is allowed to treat it like a dict/tuple
  IterableModuleKind iterableModuleKind_ = IterableModuleKind::NONE;

  // The original `nn.Module` class that we derived this ScriptModule from.
  py::object pyClass_;

  // The JIT type derived from this ConcreteModuleType.
  ClassTypePtr jitType_ = nullptr;
  // NOTE: If you ever add any more state to this struct, you need to make sure
  // operator== still makes sense! The only field that can be excluded from it
  // is `jitType_`.
};

} // namespace script
} // namespace jit
} // namespace torch
