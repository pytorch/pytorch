#include <torch/csrc/jit/script/object.h>

#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/script/compilation_unit.h>
#include <torch/csrc/jit/script/resolver.h>
#include <torch/csrc/jit/script/sugared_value.h>

namespace torch {
namespace jit {
namespace script {

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
  for (auto fn_qualname : type()->methods()) {
    if (fn_qualname.name() == basename) {
      auto maybe_fn = script::lookupMethodByQualname(_ivalue()->type(), fn_qualname);
      TORCH_INTERNAL_ASSERT(maybe_fn);
      return Method(_ivalue(), maybe_fn);
    }
  }
  return c10::nullopt;
}

void Object::define(const std::string& src, const ResolverPtr& resolver) {
  const auto self = SimpleSelf(type());
  _ivalue()->compilation_unit()->define(
      *type()->name(),
      src,
      resolver ? resolver : script::nativeResolver(),
      &self);
}

} // namespace script
} // namespace jit
} // namespace torch
