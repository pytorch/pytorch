#include "torch/csrc/jit/type.h"

#include "torch/csrc/jit/assertions.h"

#include <iostream>

namespace torch { namespace jit {

std::ostream& operator<<(std::ostream & out, const Type & t) {
  if(auto value = t.cast<TensorType>()) {
    out << at::toString(value->scalarType()) << "(";
    auto& sizes = value->sizes();
    auto& strides = value->strides();
    JIT_ASSERT(sizes.size() == strides.size());
    for (size_t i = 0; i < sizes.size(); i++) {
      if (i > 0) {
        out << ", ";
      }
      // TODO: figure out a good way to output strides, or
      // add a "debug" printing mode which adds the extra stuff
      out << sizes[i]; // << "%" << strides[i];
      int64_t expected = i + 1 < sizes.size() ? sizes[i+1]*strides[i+1] : 1;
      if (strides[i] != expected) {
        out << "!"; //mark non-contiguous
      }
    }
    out << ")";
  } else if(t.kind() == TypeKind::DynamicType) {
    out << "Dynamic";
  } else if(t.kind() == TypeKind::TupleType) {
    out << "Tuple";
  } else if(t.kind() == TypeKind::NumberType) {
    out << "Number";
  } else if(t.kind() == TypeKind::FloatType) {
    out << "float";
  } else if(t.kind() == TypeKind::IntType) {
    out << "int";
  } else if(t.kind() == TypeKind::ListType) {
    auto prim = t.cast<ListType>()->getElementType();
    out << *prim << "[]";
  } else if(t.kind() == TypeKind::NoneType) {
    out << "None";
  } else if(t.kind() == TypeKind::StringType) {
    out << "string";
  } else {
    AT_ERROR("unknown type kind");
  }
  return out;
}

DynamicTypePtr DynamicType::get() {
  static auto value = DynamicType::create();
  return value;
}
NumberTypePtr NumberType::get() {
  static auto value = NumberType::create();
  return value;
}
IntTypePtr IntType::get() {
  static auto value = IntType::create();
  return value;
}
FloatTypePtr FloatType::get() {
  static auto value = FloatType::create();
  return value;
}
NoneTypePtr NoneType::get() {
  static auto value = NoneType::create();
  return value;
}
StringTypePtr StringType::get() {
  static auto value = StringType::create();
  return value;
}
ListTypePtr ListType::ofTensors() {
  static auto value = ListType::create(DynamicType::get());
  return value;
}
ListTypePtr ListType::ofInts() {
  static auto value = ListType::create(IntType::get());
  return value;
}
ListTypePtr ListType::ofFloats() {
  static auto value = ListType::create(FloatType::get());
  return value;
}

}} // namespace torch::jit
