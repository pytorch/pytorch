#pragma once

#include <ATen/core/ATenGeneral.h>
#include <ATen/core/Generator.h>
#include <ATen/EmptyTensor.h>
#include <ATen/Formatting.h>
#include <c10/core/ScalarType.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/accumulate.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <sstream>
#include <typeinfo>
#include <numeric>
#include <memory>

#define AT_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete; \
  void operator=(const TypeName&) = delete

namespace at {

TORCH_API int _crash_if_asan(int);

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
  return expr.unsafeGetTensorImpl();
}

// Converts a TensorList (i.e. ArrayRef<Tensor> to vector of TensorImpl*)
// NB: This is ONLY used by legacy TH bindings, and ONLY used by cat.
// Once cat is ported entirely to ATen this can be deleted!
static inline std::vector<TensorImpl*> checked_dense_tensor_list_unwrap(ArrayRef<Tensor> tensors, const char * name, int pos, DeviceType device_type, ScalarType scalar_type) {
  std::vector<TensorImpl*> unwrapped;
  unwrapped.reserve(tensors.size());
  for (const auto i : c10::irange(tensors.size())) {
    const auto& expr = tensors[i];
    if (expr.layout() != Layout::Strided) {
      AT_ERROR("Expected dense tensor but got ", expr.layout(),
               " for sequence element ", i , " in sequence argument at position #", pos, " '", name, "'");
    }
    if (expr.device().type() != device_type) {
      AT_ERROR("Expected object of device type ", device_type, " but got device type ", expr.device().type(),
               " for sequence element ", i , " in sequence argument at position #", pos, " '", name, "'");
    }
    if (expr.scalar_type() != scalar_type) {
      AT_ERROR("Expected object of scalar type ", scalar_type, " but got scalar type ", expr.scalar_type(),
               " for sequence element ", i , " in sequence argument at position #", pos, " '", name, "'");
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

using at::detail::check_size_nonnegative;

namespace detail {

template <typename T>
TORCH_API
Tensor tensor_cpu(ArrayRef<T> values, const TensorOptions& options);

template <typename T>
TORCH_API
Tensor tensor_backend(ArrayRef<T> values, const TensorOptions& options);

template <typename T>
TORCH_API
Tensor tensor_complex_cpu(ArrayRef<T> values, const TensorOptions& options);

template <typename T>
TORCH_API
Tensor tensor_complex_backend(ArrayRef<T> values, const TensorOptions& options);
} // namespace detail


} // at
