#pragma once

#include "ATen/ATenGeneral.h"
#include "ATen/ArrayRef.h"
#include "ATen/Error.h"
#include "ATen/UndefinedTensor.h"

#include <algorithm>
#include <sstream>
#include <typeinfo>
#include <numeric>

#if defined(__clang__)
#define __ubsan_ignore_float_divide_by_zero__ __attribute__((no_sanitize("float-divide-by-zero")))
#define __ubsan_ignore_vptr__ __attribute__((no_sanitize("vptr")))
#else
#define __ubsan_ignore_float_divide_by_zero__
#define __ubsan_ignore_vptr__
#endif

namespace at {

AT_API int _crash_if_asan(int);

template <typename T, typename Base>
static inline T* checked_cast_storage(Base* expr, const char * name, int pos) {
  if (typeid(*expr) != typeid(T))
    AT_ERROR("Expected object of type ", T::typeString(), " but found type ", expr->type().toString(),
             " for argument #", pos, " '", name, "'");
  return static_cast<T*>(expr);
}

template <typename T, typename Base>
inline T* checked_cast_tensor(Base* expr, const char * name, int pos, bool allowNull, Backend backend, ScalarType scalar_type) {
  if(allowNull && expr == UndefinedTensor::singleton()) {
    return nullptr;
  }
  if (expr->type().backend() != backend) {
    AT_ERROR("Expected object of backend ", backend, " but got backend ", expr->type().backend(),
             " for argument #", pos, " '", name, "'");
  }
  if (expr->type().scalarType() != scalar_type) {
    AT_ERROR("Expected object of scalar type ", scalar_type, " but got scalar type ", expr->type().scalarType(),
             " for argument #", pos, " '", name, "'");
  }
  // This is going away soon; delete this when we remove the subtypes of
  // TensorImpl (so that we can eliminate the T template parameter)
  if (typeid(*expr) != typeid(T)) {
    AT_ERROR("Expected object of RTTI type ", typeid(T).name(), " but found type ", typeid(*expr).name(),
             " for argument #", pos, " '", name, "'");
  }
  return static_cast<T*>(expr);
}

// Converts a TensorList (i.e. ArrayRef<Tensor> to the underlying TH* Tensor Pointer)
template <typename T, typename TBase, typename TH>
static inline std::vector<TH*> tensor_list_checked_cast(ArrayRef<TBase> tensors, const char * name, int pos, Backend backend, ScalarType scalar_type) {
  std::vector<TH*> casted(tensors.size());
  for (unsigned int i = 0; i < tensors.size(); ++i) {
    auto *expr = tensors[i].pImpl;
    // TODO: Use the backend, scalar_type arguments to replace this
    // dynamic cast for the test
    auto result = dynamic_cast<T*>(expr);
    if (result) {
      casted[i] = result->tensor;
    } else {
      AT_ERROR("Expected a Tensor of RTTI type ", typeid(T).name(), " but found a type ", typeid(*expr).name(),
               " for sequence element ", i, " in sequence argument at position #", pos, " '", name, "'");

    }
  }
  return casted;
}

template <size_t N>
std::array<int64_t, N> check_intlist(ArrayRef<int64_t> list, const char * name, int pos, ArrayRef<int64_t> def={}) {
  if (list.empty()) {
    list = def;
  }
  auto res = std::array<int64_t, N>();
  if (list.size() == 1 && N > 1) {
    res.fill(list[0]);
    return res;
  }
  if (list.size() != N) {
    AT_ERROR("Expected a list of ", N, " ints but got ", list.size(), " for argument #", pos, " '", name, "'");
  }
  std::copy_n(list.begin(), N, res.begin());
  return res;
}

inline int64_t sum_intlist(ArrayRef<int64_t> list) {
  return std::accumulate(list.begin(), list.end(), 0);
}

inline int64_t prod_intlist(ArrayRef<int64_t> list) {
  return std::accumulate(list.begin(), list.end(), 1, std::multiplies<int64_t>());
}

} // at
