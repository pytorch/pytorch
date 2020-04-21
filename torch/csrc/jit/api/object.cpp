#include <torch/csrc/jit/api/object.h>

#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/frontend/sugared_value.h>

namespace torch {
namespace jit {

Object::Object(
    std::shared_ptr<CompilationUnit> cu,
    const c10::ClassTypePtr& type)
    : Object(c10::ivalue::Object::create(
          c10::StrongTypePtr(std::move(cu), type),
          type->numAttributes())) {}

ObjectPtr Object::_ivalue() const {
  TORCH_INTERNAL_ASSERT(_ivalue_);
  return _ivalue_;
}

c10::optional<Method> Object::find_method(const std::string& basename) const {
  for (Function* fn : type()->methods()) {
    if (fn->name() == basename) {
      return Method(_ivalue(), fn);
    }
  }
  return c10::nullopt;
}

void Object::define(const std::string& src, const ResolverPtr& resolver) {
  const auto self = SimpleSelf(type());
  _ivalue()->compilation_unit()->define(
      *type()->name(), src, resolver ? resolver : nativeResolver(), &self);
}

Object Object::copy() const {
  Object obj(_ivalue()->compilation_unit(), type());

  size_t N = type()->numAttributes();
  for (size_t i = 0; i < N; ++i) {
    IValue s = _ivalue()->getSlot(i);
    if (type()->getAttribute(i)->is_module()) {
      const Object& orig = s.toObject();
      Object copied = orig.copy();
      obj._ivalue()->setAttr(type()->getAttributeName(i), copied._ivalue());
    } else {
      obj._ivalue()->setAttr(type()->getAttributeName(i), s);
    }
  }

  return obj;
}

Object Object::deepcopy() const {
  c10::IValue::HashAliasedIValueMap memo;
  return deepcopy(memo);
}

Object Object::deepcopy(c10::IValue::HashAliasedIValueMap& memo) const {
  Object obj(_ivalue()->compilation_unit(), type());

  // Deepcopy slots. If a slot is a module - recursively copy it.
  size_t N = type()->numAttributes();
  for (size_t i = 0; i < N; ++i) {
    IValue s = _ivalue()->getSlot(i);
    obj._ivalue()->setAttr(type()->getAttributeName(i), s.deepcopy(memo));
  }

  return obj;
}

} // namespace jit
} // namespace torch
