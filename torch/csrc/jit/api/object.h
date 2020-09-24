#pragma once

#include <ATen/core/functional.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/api/method.h>

namespace torch {
namespace jit {

struct Resolver;
using ResolverPtr = std::shared_ptr<Resolver>;

using ObjectPtr = c10::intrusive_ptr<c10::ivalue::Object>;

struct TORCH_API Object {
  Object() {}
  Object(ObjectPtr _ivalue) : _ivalue_(std::move(_ivalue)) {}
  Object(std::shared_ptr<CompilationUnit> cu, const c10::ClassTypePtr& type);
  Object(
      c10::QualifiedName,
      std::shared_ptr<CompilationUnit> cu,
      bool shouldMangle = false);

  ObjectPtr _ivalue() const;

  c10::ClassTypePtr type() const {
    return _ivalue()->type();
  }

  void setattr(const std::string& name, c10::IValue v) {
    if (_ivalue()->type()->hasConstant(name)) {
      TORCH_CHECK(
          false,
          "Can't set constant '",
          name,
          "' which has value:",
          _ivalue()->type()->getConstant(name));
    } else if (auto slot = _ivalue()->type()->findAttributeSlot(name)) {
      const c10::TypePtr& expected = _ivalue()->type()->getAttribute(*slot);
      TORCH_CHECK(
          v.type()->isSubtypeOf(expected),
          "Expected a value of type '",
          expected->repr_str(),
          "' for field '",
          name,
          "', but found '",
          v.type()->repr_str(),
          "'");
      _ivalue()->setSlot(*slot, std::move(v));
    } else {
      TORCH_CHECK(false, "Module has no attribute '", name, "'");
    }
  }

  c10::IValue attr(const std::string& name) const {
    if (auto r = _ivalue()->type()->findAttributeSlot(name)) {
      return _ivalue()->getSlot(*r);
    }
    if (auto r = _ivalue()->type()->findConstantSlot(name)) {
      return _ivalue()->type()->getConstant(*r);
    }
    TORCH_CHECK(
        false,
        _ivalue()->type()->repr_str(),
        " does not have a field with name '",
        name,
        "'");
  }

  c10::IValue attr(const std::string& name, c10::IValue or_else) const {
    if (auto r = _ivalue()->type()->findAttributeSlot(name)) {
      return _ivalue()->getSlot(*r);
    }
    if (auto r = _ivalue()->type()->findConstantSlot(name)) {
      return _ivalue()->type()->getConstant(*r);
    }
    return or_else;
  }

  bool hasattr(const std::string& name) const {
    return _ivalue()->type()->hasAttribute(name) ||
        _ivalue()->type()->hasConstant(name);
  }

  // each object owns its methods. The reference returned here
  // is guaranteed to stay valid until this module has been destroyed
  Method get_method(const std::string& name) const {
    if (auto method = find_method(name)) {
      return *method;
    }
    AT_ERROR("Method '", name, "' is not defined.");
  }

  const std::vector<Method> get_methods() const {
    return c10::fmap(type()->methods(), [&](Function* func) {
      return Method(_ivalue(), func);
    });
  }

  c10::optional<Method> find_method(const std::string& basename) const;

  /// Run a method from this module.
  ///
  /// For example:
  /// @code
  ///   IValue output = module->run("relu_script", a, b);
  /// @endcode
  ///
  /// To get a compile a module from a source string, see torch::jit::compile
  ///
  /// @param method_name The name of the method to run
  /// @param args Arguments to be passed to the method
  /// @return An IValue containing the return value (or values if it is a tuple)
  /// from the method
  template <typename... Types>
  IValue run_method(const std::string& method_name, Types&&... args) {
    return get_method(method_name)({IValue(std::forward<Types>(args))...});
  }

  // so that C++ users can easily add methods
  void define(const std::string& src, const ResolverPtr& resolver = nullptr);

  size_t num_slots() const {
    return _ivalue()->slots().size();
  }

  // shallow copy the object
  Object copy() const;

  // Copies all the attributes of the object recursively without creating new
  // `ClassType`, including deepcopy of Tensors
  Object deepcopy() const;

 private:
  // mutable be we lazily initialize in module_object.
  mutable ObjectPtr _ivalue_;
};

namespace script {
// We once had a `script::` namespace that was deleted. This is for backcompat
// of the public API; new code should not use this type alias.
using Object = ::torch::jit::Object;
} // namespace script
} // namespace jit
} // namespace torch
