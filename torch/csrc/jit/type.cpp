#include "torch/csrc/jit/type.h"

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
  } else if(t.kind() == TypeKind::HandleType) {
    out << "Handle";
  } else if(t.kind() == TypeKind::DynamicType) {
    out << "Dynamic";
  } else if(t.kind() == TypeKind::TupleType) {
    out << "Tuple";
  } else {
    barf("unknown type kind");
  }
  return out;
}

TypePtr HandleType::get() {
  static auto value = std::make_shared<HandleType>();
  return value;
}
TypePtr DynamicType::get() {
  static auto value = std::make_shared<DynamicType>();
  return value;
}

}} // namespace torch::jit
