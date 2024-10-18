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
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/TransposeType.h>
#include <c10/util/StringUtil.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/allclose.h>
#include <ATen/ops/from_blob.h>
#endif
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

inline bool BlasCharToBool(char a) {
  return a == 'T' || a == 't';
}

inline at::native::TransposeType BlasCharToType(char a) {
  if (a == 'T' || a == 't') {
    return at::native::TransposeType::Transpose;
  }
  return at::native::TransposeType::NoTranspose;
}

namespace detail {

static bool NumericalCheck(ScalarType dtype, at::Tensor ref, void* other_c, int64_t size) {
  auto gpu_options = at::TensorOptions().dtype(dtype).device(at::kCUDA);
  // comparison done as 1D tensor
  at::Tensor oth = at::from_blob(other_c, {size}, gpu_options);
  at::Tensor oth_float = oth.to(at::kFloat);
  std::vector<double> atols{1e-1, 1e-2, 1e-3, 1e-4, 1e-5};
  std::vector<double> rtols{1e-1, 1e-2, 1e-3, 1e-4, 1e-5};
  double last_succeed_atol = 1;
  double last_succeed_rtol = 1;
  for (auto& atol : atols) {
    for (auto& rtol : rtols) {
      if (at::allclose(ref, oth_float, rtol, atol)) {
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

static bool NumericalCheck(ScalarType dtype, void* c, void* other_c, int64_t size) {
  auto gpu_options = at::TensorOptions().dtype(dtype).device(at::kCUDA);
  // comparison done as 1D tensor
  at::Tensor ref = at::from_blob(c,       {size}, gpu_options);
  at::Tensor ref_float = ref.to(at::kFloat);
  return NumericalCheck(dtype, ref_float, other_c, size);
}

}

template <typename T>
struct GemmParams : OpParams {
  GemmParams() {
    duplicate_inputs_ = false;
    is_reference_ = false;
  }

  std::string Signature() const override {
    return fmt::sprintf("%c%c_%ld_%ld_%ld", transa, transb, m, n, k);
  }

  int64_t _GetCountA() const {
    return lda * ((transa == 'n' || transa == 'N') ? k : m);
  }

  size_t GetSizeA() const {
    return sizeof(T) * _GetCountA();
  }

  int64_t _GetCountB() const {
    return ldb * ((transb == 'n' || transb == 'N') ? n : k);
  }

  size_t GetSizeB() const {
    return sizeof(T) * _GetCountB();
  }

  int64_t _GetCountC() const {
    return ldc * n;
  }

  size_t GetSizeC() const {
    return sizeof(T) * _GetCountC();
  }

  size_t GetSize(bool duplicate_inputs) const {
    size_t size = GetSizeC();
    if (duplicate_inputs) {
      size += GetSizeA();
      size += GetSizeB();
    }
    return size;
  }

  bool IsReferenceSupported() const {
    return true;
  }

  GemmParams* GetReference() const {
    // wrap the raw A/B/C pointers in simple 1D Tensors to transfer to CPU
    auto dtype = c10::CppTypeToScalarType<T>::value;
    auto gpu_options = at::TensorOptions().dtype(dtype).device(at::kCUDA);
    auto tensorA = at::from_blob(const_cast<T*>(a), {_GetCountA()}, gpu_options);
    auto tensorB = at::from_blob(const_cast<T*>(b), {_GetCountB()}, gpu_options);
    auto tensorC = at::from_blob(c, {_GetCountC()}, gpu_options);
    tensorA = tensorA.to(at::kCPU);
    tensorB = tensorB.to(at::kCPU);
    tensorC = tensorC.to(at::kCPU);
    if constexpr (std::is_same_v<T, c10::complex<float>> || std::is_same_v<T, c10::complex<double>>) {
      // perform cpublas on the host buffers, no cast
      at::native::cpublas::gemm_stub(
          kCPU,
          dtype,
          BlasCharToType(transa),
          BlasCharToType(transb),
          m, n, k,
          alpha,
          tensorA.data_ptr(), lda,
          tensorB.data_ptr(), ldb,
          beta,
          tensorC.data_ptr(), ldc);
    }
    else {
      // perform cpublas on the host buffers, cast all inputs to float
      float falpha = alpha;
      float fbeta = beta;
      tensorA = tensorA.to(at::kFloat);
      tensorB = tensorB.to(at::kFloat);
      tensorC = tensorC.to(at::kFloat);
      at::native::cpublas::gemm_stub(
          kCPU,
          c10::CppTypeToScalarType<float>::value,
          BlasCharToType(transa),
          BlasCharToType(transb),
          m, n, k,
          falpha,
          tensorA.data_ptr(), lda,
          tensorB.data_ptr(), ldb,
          fbeta,
          tensorC.data_ptr(), ldc);
    }
    // put the reference back on the GPU, for performance reasons
    tensorC = tensorC.to(at::kCUDA);
    // store the result in C and return
    GemmParams* copy = new GemmParams;
    *copy = *this;
    copy->is_reference_ = true;
    copy->cpu_reference_ = tensorC;
    return copy;
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
    if (is_reference_) {
      // nothing to delete
      return;
    }
    c10::cuda::CUDACachingAllocator::raw_delete(c);
    if (duplicate_inputs_) {
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<T*>(a));
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<T*>(b));
    }
  }

  TuningStatus NumericalCheck(GemmParams<T> *other) {
    auto c_dtype = c10::CppTypeToScalarType<T>::value;
    return detail::NumericalCheck(c_dtype, cpu_reference_, other->c, _GetCountC()) ? OK : FAIL;
  }

  char transa;
  char transb;
  int64_t m;
  int64_t n;
  int64_t k;
  at::opmath_type<T> alpha;
  const T* a;
  int64_t lda;
  const T* b;
  int64_t ldb;
  at::opmath_type<T> beta;
  T* c;
  int64_t ldc;
private:
  bool duplicate_inputs_;
  bool is_reference_;
  at::Tensor cpu_reference_;
};

template <typename T>
struct GemmAndBiasParams : OpParams {
  std::string Signature() const override {
    return fmt::sprintf("%c%c_%ld_%ld_%ld", transa, transb, m, n, k);
  }

  size_t GetSize(bool duplicate_inputs) const {
    size_t size = sizeof(T) * ldc * n;
    if (duplicate_inputs) {
      size += sizeof(T) * lda * ((transa == 'n' || transa == 'N') ? k : m);
      size += sizeof(T) * ldb * ((transb == 'n' || transb == 'N') ? n : k);
    }
    return size;
  }

  GemmAndBiasParams* DeepCopy(bool duplicate_inputs) const {
    GemmAndBiasParams* copy = new GemmAndBiasParams;
    *copy = *this;
    c10::DeviceIndex device = 0;
    AT_CUDA_CHECK(c10::cuda::GetDevice(&device));
    size_t c_size = ldc * n * sizeof(T);
    copy->c = static_cast<T*>(c10::cuda::CUDACachingAllocator::raw_alloc(c_size));
    AT_CUDA_CHECK(c10::cuda::CUDACachingAllocator::memcpyAsync(
        copy->c, device, c, device, c_size, getCurrentCUDAStream(device), true));
    if (duplicate_inputs) {
      size_t a_size = sizeof(T) * lda * ((transa == 'n' || transa == 'N') ? k : m);
      size_t b_size = sizeof(T) * ldb * ((transb == 'n' || transb == 'N') ? n : k);
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
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<T*>(a));
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<T*>(b));
    }
  }

  TuningStatus NumericalCheck(GemmAndBiasParams<T> *other) {
    auto c_dtype = c10::CppTypeToScalarType<T>::value;
    return detail::NumericalCheck(c_dtype, c, other->c, ldc*n) ? OK : FAIL;
  }

  char transa;
  char transb;
  int64_t m;
  int64_t n;
  int64_t k;
  at::opmath_type<T> alpha;
  const T* a;
  int64_t lda;
  const T* b;
  int64_t ldb;
  T* c;
  int64_t ldc;
  const T* bias;
  at::cuda::blas::GEMMAndBiasActivationEpilogue activation;
private:
  bool duplicate_inputs_;
};

template <typename T>
struct GemmStridedBatchedParams : OpParams {
  GemmStridedBatchedParams() {
    duplicate_inputs_ = false;
    is_reference_ = false;
  }

  std::string Signature() const override {
    return fmt::sprintf("%c%c_%ld_%ld_%ld_B_%ld", transa, transb, m, n, k, batch);
  }

  int64_t _GetCountA() const {
    return std::min(lda, stride_a) * ((transa == 'n' || transa == 'N') ? k : m) * batch;
  }

  size_t GetSizeA() const {
    return sizeof(T) * _GetCountA();
  }

  int64_t _GetCountB() const {
    return std::min(ldb, stride_b) * ((transb == 'n' || transb == 'N') ? n : k) * batch;
  }

  size_t GetSizeB() const {
    return sizeof(T) * _GetCountB();
  }

  int64_t _GetCountC() const {
    return std::min(ldc, stride_c) * n * batch;
  }

  size_t GetSizeC() const {
    return sizeof(T) * _GetCountC();
  }

  size_t GetSize(bool duplicate_inputs) const {
    size_t size = GetSizeC();
    if (duplicate_inputs) {
      size += GetSizeA();
      size += GetSizeB();
    }
    return size;
  }

  bool IsReferenceSupported() const {
    return true;
  }

  GemmStridedBatchedParams* GetReference() const {
    // wrap the raw A/B/C pointers in simple 1D Tensors to transfer to CPU
    auto dtype = c10::CppTypeToScalarType<T>::value;
    auto gpu_options = at::TensorOptions().dtype(dtype).device(at::kCUDA);
    auto tensorA = at::from_blob(const_cast<T*>(a), {_GetCountA()}, gpu_options);
    auto tensorB = at::from_blob(const_cast<T*>(b), {_GetCountB()}, gpu_options);
    auto tensorC = at::from_blob(c, {_GetCountC()}, gpu_options);
    tensorA = tensorA.to(at::kCPU);
    tensorB = tensorB.to(at::kCPU);
    tensorC = tensorC.to(at::kCPU);
    if constexpr (std::is_same_v<T, c10::complex<float>> || std::is_same_v<T, c10::complex<double>>) {
      // perform cpublas on the host buffers, no cast
      T *a = tensorA.template data_ptr<T>();
      T *b = tensorB.template data_ptr<T>();
      T *c = tensorC.template data_ptr<T>();
      for (const auto batch : c10::irange(batch)) {
        const auto a_batch = a + stride_a * batch;
        const auto b_batch = b + stride_b * batch;
        const auto c_batch = c + stride_c * batch;
        at::native::cpublas::gemm_stub(
            kCPU,
            dtype,
            BlasCharToType(transa),
            BlasCharToType(transb),
            m, n, k,
            alpha,
            a_batch, lda,
            b_batch, ldb,
            beta,
            c_batch, ldc);
      }
    }
    else {
      // perform cpublas on the host buffers, cast all inputs to float
      float falpha = alpha;
      float fbeta = beta;
      tensorA = tensorA.to(at::kFloat);
      tensorB = tensorB.to(at::kFloat);
      tensorC = tensorC.to(at::kFloat);
      float *a = tensorA.template data_ptr<float>();
      float *b = tensorB.template data_ptr<float>();
      float *c = tensorC.template data_ptr<float>();
      for (const auto batch : c10::irange(batch)) {
        const auto a_batch = a + stride_a * batch;
        const auto b_batch = b + stride_b * batch;
        const auto c_batch = c + stride_c * batch;
        at::native::cpublas::gemm_stub(
            kCPU,
            c10::CppTypeToScalarType<float>::value,
            BlasCharToType(transa),
            BlasCharToType(transb),
            m, n, k,
            falpha,
            a_batch, lda,
            b_batch, ldb,
            fbeta,
            c_batch, ldc);
      }
    }
    // put the reference back on the GPU, for performance reasons
    tensorC = tensorC.to(at::kCUDA);
    // store the result in C and return
    GemmStridedBatchedParams* copy = new GemmStridedBatchedParams;
    *copy = *this;
    copy->is_reference_ = true;
    copy->cpu_reference_ = tensorC;
    return copy;
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
      copy->a = static_cast<const T*>(c10::cuda::CUDACachingAllocator::raw_alloc(a_size));
      copy->b = static_cast<const T*>(c10::cuda::CUDACachingAllocator::raw_alloc(b_size));
      copy->duplicate_inputs_ = true;
    }
    return copy;
  }

  // only call on object returned by DeepCopy
  void Delete() {
    if (is_reference_) {
      // nothing to delete
      return;
    }
    c10::cuda::CUDACachingAllocator::raw_delete(c);
    if (duplicate_inputs_) {
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<T*>(a));
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<T*>(b));
    }
  }

  TuningStatus NumericalCheck(GemmStridedBatchedParams<T> *other) {
    auto c_dtype = c10::CppTypeToScalarType<T>::value;
    return detail::NumericalCheck(c_dtype, cpu_reference_, other->c, _GetCountC()) ? OK : FAIL;
  }

  char transa;
  char transb;
  int64_t m;
  int64_t n;
  int64_t k;
  at::opmath_type<T> alpha;
  const T* a;
  int64_t lda;
  int64_t stride_a;
  const T* b;
  int64_t ldb;
  int64_t stride_b;
  at::opmath_type<T> beta;
  T* c;
  int64_t ldc;
  int64_t stride_c;
  int64_t batch;
private:
  bool duplicate_inputs_;
  bool is_reference_;
  at::Tensor cpu_reference_;
};

template <typename T>
struct ScaledGemmParams : OpParams {
  ScaledGemmParams() {
    duplicate_inputs_ = false;
  }

  std::string Signature() const override {
    return fmt::sprintf("%c%c_%ld_%ld_%ld", transa, transb, m, n, k);
  }

  size_t GetSizeA() const {
    return sizeof(T) * lda * ((transa == 'n' || transa == 'N') ? k : m);
  }

  size_t GetSizeB() const {
    return sizeof(T) * ldb * ((transb == 'n' || transb == 'N') ? n : k);
  }

  size_t GetSizeC() const {
    return sizeof(T) * ldc * n;
  }

  size_t GetSize(bool duplicate_inputs) const {
    size_t size = GetSizeC();
    if (duplicate_inputs) {
      size += GetSizeA();
      size += GetSizeB();
    }
    return size;
  }

  bool IsReferenceSupported() const {
    return false;
  }

  ScaledGemmParams* GetReference() const {
    return nullptr;
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
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<void*>(a));
      c10::cuda::CUDACachingAllocator::raw_delete(const_cast<void*>(b));
    }
  }

  TuningStatus NumericalCheck(ScaledGemmParams<T> *other) {
    return detail::NumericalCheck(c_dtype, c, other->c, ldc*n) ? OK : FAIL;
  }

  char transa;
  char transb;
  int64_t m;
  int64_t n;
  int64_t k;
  const void* a;
  const void* a_scale_ptr;
  int64_t lda;
  ScalarType a_dtype;
  const void* b;
  const void* b_scale_ptr;
  int64_t ldb;
  ScalarType b_dtype;
  const void* bias_ptr;
  ScalarType bias_dtype;
  void* c;
  const void* c_scale_ptr;
  int64_t ldc;
  ScalarType c_dtype;
  void* amax_ptr;
  bool use_fast_accum;
private:
  bool duplicate_inputs_;
};

} // namespace at::cuda::tunable
