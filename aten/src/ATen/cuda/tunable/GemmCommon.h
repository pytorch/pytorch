// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/rocm/tunable
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Adapting TunableOp into PyTorch
// Copyright (c) Advanced Micro Devices, Inc.
//
#pragma once

#include <string>

#include <ATen/cuda/tunable/TunableOp.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/util/StringUtil.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/allclose.h>
#include <ATen/ops/from_blob.h>
#endif
#include <ATen/OpMathType.h>
#include <fmt/printf.h>

namespace at::cuda::tunable {

enum class BlasOp {
  N = 0,
  T = 1
};

inline char BlasOpToString(BlasOp op) {
  switch (op) {
    case BlasOp::N:
      return 'N';
    case BlasOp::T:
      return 'T';
  }
  TORCH_CHECK(false, "unrecognized BlasOp");
  return 'N';
}

namespace detail {

static bool NumericalCheck(ScalarType dtype, void* c, void* other_c, int64_t size) {
  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA);
  // comparison done as 1D tensor
  at::Tensor ref = at::from_blob(c,       {size}, options);
  at::Tensor oth = at::from_blob(other_c, {size}, options);
  at::Tensor ref_float = ref.to(at::kFloat);
  at::Tensor oth_float = oth.to(at::kFloat);
  std::vector<double> atols{1e-1, 1e-2, 1e-3, 1e-4, 1e-5};
  std::vector<double> rtols{1e-1, 1e-2, 1e-3, 1e-4, 1e-5};
  double last_succeed_atol = 1;
  double last_succeed_rtol = 1;
  for (auto& atol : atols) {
    for (auto& rtol : rtols) {
      if (at::allclose(ref_float, oth_float, rtol, atol)) {
        last_succeed_atol = atol;
        last_succeed_rtol = rtol;
      }
    }
  }
  if (last_succeed_atol == 1) {
    return false;
  }
  else {
    TUNABLE_LOG3("├──verify numerics: atol=", last_succeed_atol, ", rtol=", last_succeed_rtol);
  }

  return true;
}

}

// Note on GetSizeA et al.
// Tensors can be dense or arbitrarily strided. We only need our copies to be large enough.
// Our copies must be at least as large as the m n k shapes dictate, but could be larger
// depending on the lda ldb ldc values. Similarly for the batched case.

template <typename T>
struct GemmParams : OpParams {
  GemmParams() = default;

  std::string Signature() const override {
    return fmt::sprintf("%c%c_%ld_%ld_%ld", transa, transb, m, n, k);
  }

  size_t GetSizeA() const {
    size_t size_stride = lda * ((transa == 'n' || transa == 'N') ? k : m);
    size_t size_dense = m * k;
    return sizeof(T) * (size_stride > size_dense ? size_stride : size_dense);
  }

  size_t GetSizeB() const {
    size_t size_stride = ldb * ((transb == 'n' || transb == 'N') ? n : k);
    size_t size_dense = k * n;
    return sizeof(T) * (size_stride > size_dense ? size_stride : size_dense);
  }

  size_t GetSizeC() const {
    size_t size_stride = ldc * n;
    size_t size_dense = m * n;
    return sizeof(T) * (size_stride > size_dense ? size_stride : size_dense);
  }

  size_t GetSize(bool duplicate_inputs) const {
    size_t size = GetSizeC();
    if (duplicate_inputs) {
      size += GetSizeA();
      size += GetSizeB();
    }
    return size;
  }

  GemmParams* DeepCopy(bool duplicate_inputs) const {
    GemmParams* copy = new GemmParams;
    *copy = *this;
    c10::DeviceIndex device = 0;
    AT_CUDA_CHECK(c10::cuda::GetDevice(&device));
    size_t c_size = GetSizeC();
    copy->c = static_cast<T*>(c10::cuda::CUDACachingAllocator::raw_alloc(c_size));
    AT_CUDA_CHECK(c10::cuda::CUDACachingAllocator::memcpyAsync(
        copy->c, device, c, device, c_size, getCurrentCUDAStream(device), true));
    if (duplicate_inputs) {
      size_t a_size = GetSizeA();
      size_t b_size = GetSizeB();
      copy->a = static_cast<const T*>(c10::cuda::CUDACachingAllocator::raw_alloc(a_size));
      copy->b = static_cast<const T*>(c10::cuda::CUDACachingAllocator::raw_alloc(b_size));
      copy->duplicate_inputs_ = true;
    }
    return copy;
  }

  // only call on object returned by DeepCopy
  void Delete() {
    c10::cuda::CUDACachingAllocator::raw_delete(c);
    if (duplicate_inputs_) {
      // NOLINTNEXTLINE(*const-cast*)
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<T*>(a));
      // NOLINTNEXTLINE(*const-cast*)
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<T*>(b));
    }
  }

  TuningStatus NumericalCheck(GemmParams<T> *other) {
    auto c_dtype = c10::CppTypeToScalarType<T>::value;
    return detail::NumericalCheck(c_dtype, c, other->c, GetSizeC()/sizeof(T)) ? OK : FAIL;
  }

  char transa{};
  char transb{};
  int64_t m{};
  int64_t n{};
  int64_t k{};
  at::opmath_type<T> alpha;
  const T* a{};
  int64_t lda{};
  const T* b{};
  int64_t ldb{};
  at::opmath_type<T> beta;
  T* c{};
  int64_t ldc{};
private:
  bool duplicate_inputs_{false};
};

template <typename T>
struct GemmAndBiasParams : OpParams {
  std::string Signature() const override {
    return fmt::sprintf("%c%c_%ld_%ld_%ld", transa, transb, m, n, k);
  }

  size_t GetSizeA() const {
    size_t size_stride = lda * ((transa == 'n' || transa == 'N') ? k : m);
    size_t size_dense = m * k;
    return sizeof(T) * (size_stride > size_dense ? size_stride : size_dense);
  }

  size_t GetSizeB() const {
    size_t size_stride = ldb * ((transb == 'n' || transb == 'N') ? n : k);
    size_t size_dense = k * n;
    return sizeof(T) * (size_stride > size_dense ? size_stride : size_dense);
  }

  size_t GetSizeC() const {
    size_t size_stride = ldc * n;
    size_t size_dense = m * n;
    return sizeof(T) * (size_stride > size_dense ? size_stride : size_dense);
  }

  size_t GetSize(bool duplicate_inputs) const {
    size_t size = GetSizeC();
    if (duplicate_inputs) {
      size += GetSizeA();
      size += GetSizeB();
    }
    return size;
  }

  GemmAndBiasParams* DeepCopy(bool duplicate_inputs) const {
    GemmAndBiasParams* copy = new GemmAndBiasParams;
    *copy = *this;
    c10::DeviceIndex device = 0;
    AT_CUDA_CHECK(c10::cuda::GetDevice(&device));
    size_t c_size = GetSizeC();
    copy->c = static_cast<T*>(c10::cuda::CUDACachingAllocator::raw_alloc(c_size));
    AT_CUDA_CHECK(c10::cuda::CUDACachingAllocator::memcpyAsync(
        copy->c, device, c, device, c_size, getCurrentCUDAStream(device), true));
    if (duplicate_inputs) {
      size_t a_size = GetSizeA();
      size_t b_size = GetSizeB();
      copy->a = static_cast<const T*>(c10::cuda::CUDACachingAllocator::raw_alloc(a_size));
      copy->b = static_cast<const T*>(c10::cuda::CUDACachingAllocator::raw_alloc(b_size));
      copy->duplicate_inputs_ = true;
    }
    return copy;
  }

  // only call on object returned by DeepCopy
  void Delete() {
    c10::cuda::CUDACachingAllocator::raw_delete(c);
    if (duplicate_inputs_) {
      // NOLINTNEXTLINE(*const-cast)
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<T*>(a));
      // NOLINTNEXTLINE(*const-cast)
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<T*>(b));
    }
  }

  TuningStatus NumericalCheck(GemmAndBiasParams<T> *other) {
    auto c_dtype = c10::CppTypeToScalarType<T>::value;
    return detail::NumericalCheck(c_dtype, c, other->c, GetSizeC()/sizeof(T)) ? OK : FAIL;
  }

  char transa{};
  char transb{};
  int64_t m{};
  int64_t n{};
  int64_t k{};
  at::opmath_type<T> alpha{};
  const T* a{};
  int64_t lda{};
  const T* b{};
  int64_t ldb{};
  T* c{};
  int64_t ldc{};
  const T* bias{};
  at::cuda::blas::GEMMAndBiasActivationEpilogue activation{};
private:
  bool duplicate_inputs_{false};
};

template <typename T>
struct GemmStridedBatchedParams : OpParams {
  std::string Signature() const override {
    return fmt::sprintf("%c%c_%ld_%ld_%ld_B_%ld", transa, transb, m, n, k, batch);
  }

  size_t GetSizeA() const {
    size_t size_stride = stride_a * batch;
    size_t size_dense = m * k * batch;
    return sizeof(T) * (size_stride > size_dense ? size_stride : size_dense);
  }

  size_t GetSizeB() const {
    size_t size_stride = stride_b * batch;
    size_t size_dense = k * n * batch;
    return sizeof(T) * (size_stride > size_dense ? size_stride : size_dense);
  }

  size_t GetSizeC() const {
    size_t size_stride = stride_c * batch;
    size_t size_dense = m * n * batch;
    return sizeof(T) * (size_stride > size_dense ? size_stride : size_dense);
  }

  size_t GetSize(bool duplicate_inputs) const {
    size_t size = GetSizeC();
    if (duplicate_inputs) {
      size += GetSizeA();
      size += GetSizeB();
    }
    return size;
  }

  GemmStridedBatchedParams* DeepCopy(bool duplicate_inputs) const {
    GemmStridedBatchedParams* copy = new GemmStridedBatchedParams;
    *copy = *this;
    c10::DeviceIndex device = 0;
    AT_CUDA_CHECK(c10::cuda::GetDevice(&device));
    size_t c_size = GetSizeC();
    copy->c = static_cast<T*>(c10::cuda::CUDACachingAllocator::raw_alloc(c_size));
    AT_CUDA_CHECK(c10::cuda::CUDACachingAllocator::memcpyAsync(
        copy->c, device, c, device, c_size, getCurrentCUDAStream(device), true));
    if (duplicate_inputs) {
      size_t a_size = GetSizeA();
      size_t b_size = GetSizeB();
      // NOLINTNEXTLINE(*const-cast*)
      copy->a = static_cast<const T*>(c10::cuda::CUDACachingAllocator::raw_alloc(a_size));
      // NOLINTNEXTLINE(*const-cast*)
      copy->b = static_cast<const T*>(c10::cuda::CUDACachingAllocator::raw_alloc(b_size));
      copy->duplicate_inputs_ = true;
    }
    return copy;
  }

  // only call on object returned by DeepCopy
  void Delete() {
    c10::cuda::CUDACachingAllocator::raw_delete(c);
    if (duplicate_inputs_) {
      // NOLINTNEXTLINE(*const-cast*)
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<T*>(a));
      // NOLINTNEXTLINE(*const-cast*)
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<T*>(b));
    }
  }

  TuningStatus NumericalCheck(GemmStridedBatchedParams<T> *other) {
    auto c_dtype = c10::CppTypeToScalarType<T>::value;
    return detail::NumericalCheck(c_dtype, c, other->c, GetSizeC()/sizeof(T)) ? OK : FAIL;
  }

  char transa{};
  char transb{};
  int64_t m{};
  int64_t n{};
  int64_t k{};
  at::opmath_type<T> alpha{};
  const T* a{};
  int64_t lda{};
  int64_t stride_a{};
  const T* b{};
  int64_t ldb{};
  int64_t stride_b{};
  at::opmath_type<T> beta;
  T* c{};
  int64_t ldc{};
  int64_t stride_c{};
  int64_t batch{};
private:
  bool duplicate_inputs_{false};
};

template <typename T>
struct ScaledGemmParams : OpParams {
  ScaledGemmParams() = default;

  std::string Signature() const override {
    return fmt::sprintf("%c%c_%ld_%ld_%ld", transa, transb, m, n, k);
  }

  size_t GetSizeA() const {
    size_t size_stride = lda * ((transa == 'n' || transa == 'N') ? k : m);
    size_t size_dense = m * k;
    return sizeof(T) * (size_stride > size_dense ? size_stride : size_dense);
  }

  size_t GetSizeB() const {
    size_t size_stride = ldb * ((transb == 'n' || transb == 'N') ? n : k);
    size_t size_dense = k * n;
    return sizeof(T) * (size_stride > size_dense ? size_stride : size_dense);
  }

  size_t GetSizeC() const {
    size_t size_stride = ldc * n;
    size_t size_dense = m * n;
    return sizeof(T) * (size_stride > size_dense ? size_stride : size_dense);
  }

  size_t GetSize(bool duplicate_inputs) const {
    size_t size = GetSizeC();
    if (duplicate_inputs) {
      size += GetSizeA();
      size += GetSizeB();
    }
    return size;
  }

  ScaledGemmParams* DeepCopy(bool duplicate_inputs) const {
    ScaledGemmParams* copy = new ScaledGemmParams;
    *copy = *this;
    c10::DeviceIndex device = 0;
    AT_CUDA_CHECK(c10::cuda::GetDevice(&device));
    size_t c_size = GetSizeC();
    copy->c = c10::cuda::CUDACachingAllocator::raw_alloc(c_size);
    AT_CUDA_CHECK(c10::cuda::CUDACachingAllocator::memcpyAsync(
        copy->c, device, c, device, c_size, getCurrentCUDAStream(device), true));
    if (duplicate_inputs) {
      size_t a_size = GetSizeA();
      size_t b_size = GetSizeB();
      copy->a = c10::cuda::CUDACachingAllocator::raw_alloc(a_size);
      copy->b = c10::cuda::CUDACachingAllocator::raw_alloc(b_size);
      copy->duplicate_inputs_ = true;
    }
    return copy;
  }

  // only call on object returned by DeepCopy
  void Delete() {
    c10::cuda::CUDACachingAllocator::raw_delete(c);
    if (duplicate_inputs_) {
      // NOLINTNEXTLINE(*const-cast*)
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<void*>(a));
      // NOLINTNEXTLINE(*const-cast*)
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<void*>(b));
    }
  }

  TuningStatus NumericalCheck(ScaledGemmParams<T> *other) {
    return detail::NumericalCheck(c_dtype, c, other->c, GetSizeC()/sizeof(T)) ? OK : FAIL;
  }

  char transa{};
  char transb{};
  int64_t m{};
  int64_t n{};
  int64_t k{};
  const void* a{};
  const void* a_scale_ptr{};
  int64_t lda{};
  ScalarType a_dtype{};
  const void* b{};
  const void* b_scale_ptr{};
  int64_t ldb{};
  ScalarType b_dtype{};
  const void* bias_ptr{};
  ScalarType bias_dtype{};
  void* c{};
  const void* c_scale_ptr{};
  int64_t ldc{};
  ScalarType c_dtype{};
  void* amax_ptr{};
  bool use_fast_accum{};
private:
  bool duplicate_inputs_{false};
};

} // namespace at::cuda::tunable
