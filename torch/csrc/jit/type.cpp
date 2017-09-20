#include "torch/csrc/jit/type.h"

#include <iostream>

namespace torch { namespace jit {

std::ostream& operator<<(std::ostream & out, const Type & t) {
  TYPE_IF(&t, MultiType)
    out << "Multi";
  TYPE_ELSEIF(HandleType)
    out << "Handle";
  TYPE_ELSEIF(TensorType)
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
  TYPE_END()
  return out;
}

}} // namespace torch::jit
