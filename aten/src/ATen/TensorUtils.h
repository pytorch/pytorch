#pragma once

#include <ATen/DimVector.h>
#include <ATen/EmptyTensor.h>
#include <ATen/Tensor.h>
#include <ATen/TensorGeometry.h>
#include <ATen/Utils.h>

#include <utility>

// These functions are NOT in Utils.h, because this file has a dep on Tensor.h

#define TORCH_CHECK_TENSOR_ALL(cond, ...) \
  TORCH_CHECK((cond)._is_all_true().item<bool>(), __VA_ARGS__);

namespace at {

// The following are utility functions for checking that arguments
// make sense.  These are particularly useful for native functions,
// which do NO argument checking by default.

struct TORCH_API TensorArg {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const Tensor& tensor;
  const char* name;
  int pos; // 1-indexed
  TensorArg(const Tensor& tensor, const char* name, int pos)
      : tensor(tensor), name(name), pos(pos) {}
  // Try to mitigate any possibility of dangling reference to temporaries.
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  TensorArg(Tensor&& tensor, const char* name, int pos) = delete;
  const Tensor* operator->() const {
    return &tensor;
  }
  const Tensor& operator*() const {
    return tensor;
  }
};

struct TORCH_API TensorGeometryArg {
  TensorGeometry tensor;
  const char* name;
  int pos; // 1-indexed
  /* implicit */ TensorGeometryArg(TensorArg arg)
      : tensor(TensorGeometry{arg.tensor}), name(arg.name), pos(arg.pos) {}
  TensorGeometryArg(TensorGeometry tensor, const char* name, int pos)
      : tensor(std::move(tensor)), name(name), pos(pos) {}
  const TensorGeometry* operator->() const {
    return &tensor;
  }
  const TensorGeometry& operator*() const {
    return tensor;
  }
};

// A string describing which function did checks on its input
// arguments.
// TODO: Consider generalizing this into a call stack.
using CheckedFrom = const char*;

// The undefined convention: singular operators assume their arguments
// are defined, but functions which take multiple tensors will
// implicitly filter out undefined tensors (to make it easier to perform
// tests which should apply if the tensor is defined, and should not
// otherwise.)
//
// NB: This means that the n-ary operators take lists of TensorArg,
// not TensorGeometryArg, because the Tensor to TensorGeometry
// conversion will blow up if you have undefined tensors.

TORCH_API std::ostream& operator<<(
    std::ostream& out,
    const TensorGeometryArg& t);
TORCH_API void checkDim(
    CheckedFrom c,
    const Tensor& tensor,
    const char* name,
    int pos, // 1-indexed
    int64_t dim);
TORCH_API void checkDim(CheckedFrom c, const TensorGeometryArg& t, int64_t dim);
// NB: this is an inclusive-exclusive range
TORCH_API void checkDimRange(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t dim_start,
    int64_t dim_end);
TORCH_API void checkSameDim(
    CheckedFrom c,
    const TensorGeometryArg& t1,
    const TensorGeometryArg& t2);
TORCH_API void checkContiguous(CheckedFrom c, const TensorGeometryArg& t);
TORCH_API void checkAllContiguous(CheckedFrom c, at::ArrayRef<TensorArg> ts);
TORCH_API void checkSize(
    CheckedFrom c,
    const TensorGeometryArg& t,
    IntArrayRef sizes);
TORCH_API void checkSize_symint(
    CheckedFrom c,
    const TensorGeometryArg& t,
    c10::SymIntArrayRef sizes);
TORCH_API void checkSize(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t dim,
    int64_t size);
TORCH_API void checkSize_symint(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t dim,
    const c10::SymInt& size);
TORCH_API void checkNumel(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t numel);
TORCH_API void checkSameNumel(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);
TORCH_API void checkAllSameNumel(CheckedFrom c, ArrayRef<TensorArg> tensors);
TORCH_API void checkScalarType(CheckedFrom c, const TensorArg& t, ScalarType s);
TORCH_API void checkScalarTypes(
    CheckedFrom c,
    const TensorArg& t,
    at::ArrayRef<ScalarType> l);
TORCH_API void checkSameGPU(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);
TORCH_API void checkAllSameGPU(CheckedFrom c, ArrayRef<TensorArg> tensors);
TORCH_API void checkSameType(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);
TORCH_API void checkAllSameType(CheckedFrom c, ArrayRef<TensorArg> tensors);
TORCH_API void checkSameSize(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);
TORCH_API void checkAllSameSize(CheckedFrom c, ArrayRef<TensorArg> tensors);
TORCH_API void checkDefined(CheckedFrom c, const TensorArg& t);
TORCH_API void checkAllDefined(CheckedFrom c, at::ArrayRef<TensorArg> t);

// FixMe: does TensorArg slow things down?
TORCH_API void checkBackend(
    CheckedFrom c,
    at::ArrayRef<Tensor> t,
    at::Backend backend);

TORCH_API void checkDeviceType(
    CheckedFrom c,
    at::ArrayRef<Tensor> tensors,
    at::DeviceType device_type);

TORCH_API void checkLayout(CheckedFrom c, const Tensor& t, Layout layout);

TORCH_API void checkLayout(
    CheckedFrom c,
    at::ArrayRef<Tensor> tensors,
    at::Layout layout);

// Methods for getting data_ptr if tensor is defined
TORCH_API void* maybe_data_ptr(const Tensor& tensor);
TORCH_API void* maybe_data_ptr(const TensorArg& tensor);

TORCH_API void check_dim_size(
    const Tensor& tensor,
    int64_t dim,
    int64_t dim_size,
    int64_t size);

namespace detail {
TORCH_API std::vector<int64_t> defaultStrides(IntArrayRef sizes);

TORCH_API c10::optional<std::vector<int64_t>> computeStride(
    IntArrayRef oldshape,
    IntArrayRef oldstride,
    IntArrayRef newshape);

TORCH_API c10::optional<SymDimVector> computeStride(
    c10::SymIntArrayRef oldshape,
    c10::SymIntArrayRef oldstride,
    c10::SymIntArrayRef newshape);

TORCH_API c10::optional<DimVector> computeStride(
    IntArrayRef oldshape,
    IntArrayRef oldstride,
    const DimVector& newshape);

} // namespace detail
} // namespace at
