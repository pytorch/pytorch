#pragma once
#include <torch/csrc/jit/api/api.h>


namespace torch {
namespace jit {
namespace api {



bool Type::isSubtypeOf(const Type& rhs) {
 return type_->isSubtypeOf(rhs.type_);
}
std::string Type::python_str() const noexcept {
  return type_->python_str();
}
const Type Type::Bool() {
  return Type(c10::BoolType::get());
}
const Type Type::DeviceObject() {
  return Type(c10::DeviceObjType::get());
}
const Type Type::Float() {
  return Type(c10::FloatType::get());
}
const Type Type::Int() {
  return Type(c10::IntType::get());
}
const Type Type::Layout() {
  return Type(c10::LayoutType::get());
}
const Type Type::None() {
  return Type(c10::NoneType::get());
}
const Type Type::String() {
  return Type(c10::StringType::get());
}
const Type Type::Tensor() {
  return Type(c10::TensorType::get());
}

Type ClassType::attr(const std::string& name) {
  auto attr = type_->expect<c10::ClassType>()->findAttribute(name);
  TORCH_CHECK(attr != nullptr, "Did not find attribute " + name);
  return Type(attr);
}
bool ClassType::hasattr(const std::string& name) {
  auto attr = type_->expect<c10::ClassType>()->findAttribute(name);
  return attr != nullptr;
}
iterable<Field<Type>> ClassType::attributes() {
  // type_->expect<c10::ClassType>()->attributeNames();
  return class_attribute_type_iterator(type_->expect<c10::ClassType>(), 0);
}

} // namespace api
} // namespace jit
} // namespace torch
