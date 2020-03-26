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
      *type()->name(),
      src,
      resolver ? resolver : nativeResolver(),
      &self);
}

} // namespace jit
} // namespace torch
