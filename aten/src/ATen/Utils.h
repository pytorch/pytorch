#pragma once

#include <ATen/core/ATenGeneral.h>
#include <ATen/core/Generator.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/UndefinedTensorImpl.h>

#include <c10/core/ScalarType.h>
#include <ATen/Formatting.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <sstream>
#include <typeinfo>
#include <numeric>
#include <memory>

#if defined(__clang__)
#define __ubsan_ignore_float_divide_by_zero__ __attribute__((no_sanitize("float-divide-by-zero")))
#define __ubsan_ignore_vptr__ __attribute__((no_sanitize("vptr")))
#else
#define __ubsan_ignore_float_divide_by_zero__
#define __ubsan_ignore_vptr__
#endif

#define AT_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete; \
  void operator=(const TypeName&) = delete

namespace at {

CAFFE2_API int _crash_if_asan(int);

static inline const Storage& checked_storage(
    const Storage& expr,
    const char* name,
    int pos,
    DeviceType device_type,
    caffe2::TypeMeta dtype) {
  if (expr.device_type() != device_type) {
    AT_ERROR(
        "Expected object of device type ",
        device_type,
        " but got device type ",
        expr.data_ptr().device().type(),
        " for argument #",
        pos,
        " '",
        name,
        "'");
  }
  if (expr.dtype() != dtype) {
    AT_ERROR(
        "Expected object of data type ",
        dtype,
        " but got data type ",
        expr.dtype().id(),
        " for argument #",
        pos,
        " '",
        name,
        "'");
  }
  return expr;
}

// TODO: This unwrapping code is ONLY used for TH bindings; once TH goes
// away, we can delete this function
static inline TensorImpl* checked_dense_tensor_unwrap(const Tensor& expr, const char * name, int pos, const char * api, bool allowNull, DeviceType device_type, ScalarType scalar_type) {
  if(allowNull && !expr.defined()) {
    return nullptr;
  }
  if (expr.layout() != Layout::Strided) {
    AT_ERROR("Expected dense tensor but got ", expr.layout(),
             " for argument #", pos, " '", name, "' in call to ", api);
  }
  if (expr.device().type() != device_type) {
    AT_ERROR("Expected object of device type ", device_type, " but got device type ", expr.device().type(),
             " for argument #", pos, " '", name, "' in call to ", api);
  }
  if (expr.scalar_type() != scalar_type) {
    AT_ERROR("Expected object of scalar type ", scalar_type, " but got scalar type ", expr.scalar_type(),
             " for argument #", pos, " '", name, "' in call to ", api);
  }
  if (expr.is_variable()) {  // TODO: change this to check `.requires_grad()` and `GradMode::is_enabled()` when Variable and Tensor are merged
    AT_ERROR("Expected Tensor (not Variable) for argument #", pos, " '", name, "' in call to ", api);
  }
  return expr.unsafeGetTensorImpl();
}

// Converts a TensorList (i.e. ArrayRef<Tensor> to vector of TensorImpl*)
static inline std::vector<TensorImpl*> checked_tensor_list_unwrap(ArrayRef<Tensor> tensors, const char * name, int pos, Backend backend, ScalarType scalar_type) {
  std::vector<TensorImpl*> unwrapped;
  unwrapped.reserve(tensors.size());
  for (unsigned int i = 0; i < tensors.size(); ++i) {
    const auto& expr = tensors[i];
    if (tensorTypeIdToBackend(impl::dispatchTypeId(expr.type_set())) != backend) {
      AT_ERROR("Expected object of backend ", backend, " but got backend ", tensorTypeIdToBackend(impl::dispatchTypeId(expr.type_set())),
               " for sequence element ", i, " in sequence argument at position #", pos, " '", name, "'");
    }
    if (expr.scalar_type() != scalar_type) {
      AT_ERROR("Expected object of scalar type ", scalar_type, " but got scalar type ", expr.scalar_type(),
               " for sequence element ", i , " in sequence argument at position #", pos, " '", name, "'");
    }
    if (expr.is_variable()) {  // TODO: change this to check `.requires_grad()` and `GradMode::is_enabled()` when Variable and Tensor are merged
      AT_ERROR("Expected Tensor (not Variable) for sequence element ",
               i , " in sequence argument at position #", pos, " '", name, "'");
    }
    unwrapped.emplace_back(expr.unsafeGetTensorImpl());
  }
  return unwrapped;
}

template <size_t N>
std::array<int64_t, N> check_intlist(ArrayRef<int64_t> list, const char * name, int pos) {
  if (list.empty()) {
    // TODO: is this necessary?  We used to treat nullptr-vs-not in IntList differently
    // with strides as a way of faking optional.
    list = {};
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
  return std::accumulate(list.begin(), list.end(), 0ll);
}

inline int64_t prod_intlist(ArrayRef<int64_t> list) {
  return std::accumulate(list.begin(), list.end(), 1ll, std::multiplies<int64_t>());
}

/**
 * Utility function used in tensor implementations, which
 * supplies the default generator to tensors, if an input generator
 * is not supplied. The input Generator* is also static casted to
 * the backend generator type (CPU/CUDAGenerator etc.)
 */
template <typename T>
static inline T * get_generator_or_default(Generator * expr, Generator * defaultValue) {
  if (!expr) {
    expr = defaultValue;
  }
  if (T::device_type() == expr->device().type()) {
    return static_cast<T*>(expr);
  }
  AT_ERROR("Expected a '", T::device_type(), "' device type for generator but found '", expr->device().type(), "'");
}

/**
 * Utility function to static cast input Generator* to
 * the backend generator type (CPU/CUDAGenerator etc.)
 */
template <typename T>
static inline T * check_generator(Generator * expr) {
  if (T::device_type() == expr->device().type()) {
    return static_cast<T*>(expr);
  }
  AT_ERROR("Expected a '", T::device_type(), "' device type for generator but found '", expr->device().type(), "'");
}

} // at
