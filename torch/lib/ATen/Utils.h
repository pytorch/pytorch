#pragma once

#include "ArrayRef.h"
#include <sstream>

namespace at {

#define AT_ASSERT(cond, ...) if (! (cond) ) { at::runtime_error(__VA_ARGS__); }

[[noreturn]]
void runtime_error(const char *format, ...);

template <typename T, typename Base>
static inline T* checked_cast(Base* expr, const char * name, int pos, bool allowNull) {
  if(!expr) {
    if (allowNull) {
      return (T*) expr;
    }
    runtime_error("Expected a Tensor of type %s but found an undefined Tensor for argument #%d '%s'",
      T::typeString(),pos,name);
  }
  if(auto result = dynamic_cast<T*>(expr))
    return result;
  runtime_error("Expected object of type %s but found type %s for argument #%d '%s'",
    T::typeString(),expr->type().toString(),pos,name);
}

// Converts a TensorList (i.e. ArrayRef<Tensor> to the underlying TH* Tensor Pointer)
template <typename T, typename TBase, typename TH>
static inline std::vector<TH*> tensor_list_checked_cast(ArrayRef<TBase> tensors, const char * name, int pos) {
  std::vector<TH*> casted(tensors.size());
  for (unsigned int i = 0; i < tensors.size(); ++i) {
    auto *expr = tensors[i].pImpl;
    if (!expr) {
      runtime_error("Expected a Tensor of type %s but found an undefined Tensor for sequence element %u "
                    " in sequence argument at position #%d '%s'",
                    T::typeString(),i,pos,name);
    }
    auto result = dynamic_cast<T*>(expr);
    if (result) {
      casted[i] = result->tensor;
    } else {
      runtime_error("Expected a Tensor of type %s but found a type %s for sequence element %u "
                    " in sequence argument at position #%d '%s'",
                    T::typeString(),expr->type().toString(),i,pos,name);

    }
  }
  return casted;
}

static inline int64_t maybe_wrap_dim(int64_t dim, int64_t dim_post_expr) {
  int64_t corrected_dim = std::max<int64_t>(dim_post_expr, 0);
  if (corrected_dim <= 0) {
    std::ostringstream oss;
    oss << "dimension specified as " << dim << " but tensor has no dimensions";
    throw std::runtime_error(oss.str());
  }
  if (dim < -(corrected_dim) || dim >= (corrected_dim)) {
    std::ostringstream oss;
    oss << "dimension out of range (expected to be in range of [" << -(corrected_dim)
        << ", " << (corrected_dim)-1 << "], but got " << dim << ")",
    throw std::runtime_error(oss.str());
  }
  if (dim  < 0) dim += corrected_dim;
  return dim;
}

} // at
