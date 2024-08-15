#pragma once

#include <ATen/OpMathType.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TransposeType.h>
#include <c10/util/complex.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Scalar.h>

#include <ATen/Config.h>
#if AT_MKLDNN_ENABLED() && (defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC)))
#include <oneapi/dnnl/dnnl_ukernel.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#endif

namespace at::native::cpublas {

namespace internal {
void normalize_last_dims(
  TransposeType transa, TransposeType transb,
  int64_t m, int64_t n, int64_t k,
  int64_t *lda, int64_t *ldb, int64_t *ldc);
}  // namespace internal

using gemm_fn = void(*)(
    at::ScalarType type,
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const Scalar& alpha,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const Scalar& beta,
    void *c, int64_t ldc);

DECLARE_DISPATCH(gemm_fn, gemm_stub);

template <typename scalar_t>
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    at::opmath_type<scalar_t> alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    at::opmath_type<scalar_t> beta,
    scalar_t *c, int64_t ldc) {
  internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
  gemm_stub(
    kCPU, c10::CppTypeToScalarType<scalar_t>::value,
    transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    double alpha,
    const double *a, int64_t lda,
    const double *b, int64_t ldb,
    double beta,
    double *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const float *a, int64_t lda,
    const float *b, int64_t ldb,
    float beta,
    float *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const at::BFloat16 *a, int64_t lda,
    const at::BFloat16 *b, int64_t ldb,
    float beta,
    at::BFloat16 *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const at::BFloat16 *a, int64_t lda,
    const at::BFloat16 *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const at::Half *a, int64_t lda,
    const at::Half *b, int64_t ldb,
    float beta,
    at::Half *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const at::Half *a, int64_t lda,
    const at::Half *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    c10::complex<double> alpha,
    const c10::complex<double> *a, int64_t lda,
    const c10::complex<double> *b, int64_t ldb,
    c10::complex<double> beta,
    c10::complex<double> *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    c10::complex<float> alpha,
    const c10::complex<float> *a, int64_t lda,
    const c10::complex<float> *b, int64_t ldb,
    c10::complex<float> beta,
    c10::complex<float> *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    int64_t alpha,
    const int64_t *a, int64_t lda,
    const int64_t *b, int64_t ldb,
    int64_t beta,
    int64_t *c, int64_t ldc);

template <typename scalar_t>
void gemm_batched(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t * const *a, int64_t lda,
    const scalar_t * const *b, int64_t ldb,
    const scalar_t beta,
    scalar_t * const *c, int64_t ldc);

template <typename scalar_t>
void gemm_batched_with_stride(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda, int64_t batch_stride_a,
    const scalar_t *b, int64_t ldb, int64_t batch_stride_b,
    scalar_t beta,
    scalar_t *c, int64_t ldc, int64_t batch_stride_c);

using axpy_fn = void(*)(at::ScalarType type, int64_t n, const Scalar& a, const void *x, int64_t incx, void *y, int64_t incy);

DECLARE_DISPATCH(axpy_fn, axpy_stub);

template<typename scalar_t>
void axpy(int64_t n, scalar_t a, const scalar_t *x, int64_t incx, scalar_t *y, int64_t incy){
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  axpy_stub(
      kCPU, c10::CppTypeToScalarType<scalar_t>::value,
      n, a, x, incx, y, incy);
}

void axpy(int64_t n, double a, const double *x, int64_t incx, double *y, int64_t incy);
void axpy(int64_t n, float a, const float *x, int64_t incx, float *y, int64_t incy);
void axpy(int64_t n, c10::complex<double> a, const c10::complex<double> *x, int64_t incx, c10::complex<double> *y, int64_t incy);
void axpy(int64_t n, c10::complex<float> a, const c10::complex<float> *x, int64_t incx, c10::complex<float> *y, int64_t incy);

using copy_fn = void(*)(at::ScalarType type, int64_t n, const void *x, int64_t incx, void *y, int64_t incy);

DECLARE_DISPATCH(copy_fn, copy_stub);

template<typename scalar_t>
void copy(int64_t n, const scalar_t *x, int64_t incx, scalar_t *y, int64_t incy) {
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  copy_stub(
      kCPU, c10::CppTypeToScalarType<scalar_t>::value,
      n, x, incx, y, incy);
}

void copy(int64_t n, const double *x, int64_t incx, double *y, int64_t incy);
void copy(int64_t n, const float *x, int64_t incx, float *y, int64_t incy);
void copy(int64_t n, const c10::complex<double> *x, int64_t incx, c10::complex<double> *y, int64_t incy);
void copy(int64_t n, const c10::complex<float> *x, int64_t incx, c10::complex<float> *y, int64_t incy);

#if AT_MKLDNN_ENABLED() && (defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC)))
template <typename key_t, typename value_t>
struct Kernel_Cache {
  using kstore_t = std::unordered_map<key_t, std::shared_ptr<value_t>>;
  static inline std::shared_ptr<value_t>&& fetch_or_create(
      const key_t& key,
      const std::function<std::shared_ptr<value_t>()>& callback) {
    auto&& search = get_store().find(key);
    if (search != get_store().end()) {
      return std::move(search->second);
    } else {
      get_store().insert({key, callback()});
      return std::move(get_store()[key]);
    }
  }

  static inline kstore_t& get_store() {
    static thread_local kstore_t cache_kernels;
    return cache_kernels;
  }
};

inline dnnl::memory::data_type get_dnnl_dtype(ScalarType dtype) {
  if (dtype == ScalarType::Float) {
    return dnnl::memory::data_type::f32;
  } else if (dtype == ScalarType::BFloat16) {
    return dnnl::memory::data_type::bf16;
  } else if (dtype == ScalarType::Half) {
    return dnnl::memory::data_type::f16;
  } else if (dtype == ScalarType::Byte) {
    return dnnl::memory::data_type::u8;
  } else if (dtype == ScalarType::Char) {
    return dnnl::memory::data_type::s8;
  } else {
    TORCH_CHECK(false, "get_dnnl_dtype expects float/bfloat16/half/int8 tensor input");
  }
}
#endif

struct BrgemmKey {
  int64_t M;
  int64_t N;
  int64_t K;
  int64_t batch_size;
  int64_t lda;
  int64_t ldb;
  int64_t ldc;
  ScalarType dt_a;
  ScalarType dt_b;
  ScalarType dt_c;
  float alpha;
  float beta;
  BrgemmKey(
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t batch_size,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      ScalarType dt_a,
      ScalarType dt_b,
      ScalarType dt_c,
      float alpha,
      float beta)
      : M(M),
        N(N),
        K(K),
        batch_size(batch_size),
        lda(lda),
        ldb(ldb),
        ldc(ldc),
        dt_a(dt_a),
        dt_b(dt_b),
        dt_c(dt_c),
        alpha(alpha),
        beta(beta) {}
  bool operator==(const BrgemmKey& other) const {
    return M == other.M && N == other.N && K == other.K &&
        batch_size == other.batch_size && lda == other.lda &&
        ldb == other.ldb && ldc == other.ldc && dt_a == other.dt_a &&
        dt_b == other.dt_b && dt_c == other.dt_c && alpha == other.alpha &&
        beta == other.beta;
  }
};

struct PackKey {
  int64_t K;
  int64_t N;
  int64_t ld_in;
  int64_t ld_out;
  ScalarType dt_in;
  ScalarType dt_out;
  PackKey(
      int64_t K,
      int64_t N,
      int64_t ld_in,
      int64_t ld_out,
      ScalarType dt_in,
      ScalarType dt_out)
      : K(K),
        N(N),
        ld_in(ld_in),
        ld_out(ld_out),
        dt_in(dt_in),
        dt_out(dt_out) {}
  bool operator==(const PackKey& other) const {
    return N == other.N && K == other.K && ld_in == other.ld_in &&
        ld_out == other.ld_out && dt_in == other.dt_in &&
        dt_out == other.dt_out;
  }
};

#if AT_MKLDNN_ENABLED() && (defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC)))
// Helper struct for convenient brgemm configuration
struct GemmHelper {
  GemmHelper(
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t bs,
      int64_t ld_a,
      int64_t ld_b,
      int64_t ld_c,
      ScalarType dt_a,
      ScalarType dt_b,
      ScalarType dt_c,
      const float alpha,
      const float beta) {
    // Create brgemm
    brg = dnnl::ukernel::brgemm(
        M,
        N,
        K,
        bs,
        ld_a,
        ld_b,
        ld_c,
        get_dnnl_dtype(dt_a),
        get_dnnl_dtype(dt_b),
        get_dnnl_dtype(dt_c),
        alpha,
        beta);
    // Create a scratchpad buffer for the brgemm execution
    scratchpad = std::vector<uint8_t>(brg.get_scratchpad_size());
    // Prepare default vector of pairs of tensors A and B offsets for each batch.
    A_B_offsets.reserve(1);
    A_B_offsets[0] = std::make_pair(0, 0);
  }
  dnnl::ukernel::brgemm brg;
  std::vector<uint8_t> scratchpad;
  std::vector<std::pair<int64_t, int64_t>> A_B_offsets;
};

struct Brgemm : public Kernel_Cache<BrgemmKey, GemmHelper> {
  // Fetch/create GemmHelper object and execute brgemm with batch size and
  // extern offset
  template <typename scalar_t_a, typename scalar_t_b, typename scalar_t_c>
  static inline void call(
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t bs,
      int64_t ld_a,
      int64_t ld_b,
      int64_t ld_c,
      const float alpha,
      const float beta,
      const scalar_t_a* A,
      const scalar_t_b* B,
      const std::vector<std::pair<int64_t, int64_t>>& offsets,
      scalar_t_c* C) {
    auto&& key = BrgemmKey(
        M,
        N,
        K,
        bs,
        ld_a,
        ld_b,
        ld_c,
        c10::CppTypeToScalarType<scalar_t_a>::value,
        c10::CppTypeToScalarType<scalar_t_b>::value,
        c10::CppTypeToScalarType<scalar_t_c>::value,
        alpha,
        beta);
    // Fetch/create GemmHelper object
    auto&& value = fetch_or_create(key, [&]() {
      auto&& v = std::make_shared<GemmHelper>(
          M,
          N,
          K,
          1,
          ld_a,
          ld_b,
          ld_c,
          c10::CppTypeToScalarType<scalar_t_a>::value,
          c10::CppTypeToScalarType<scalar_t_b>::value,
          c10::CppTypeToScalarType<scalar_t_c>::value,
          alpha,
          beta);
      (*v).brg.generate();
      return std::move(v);
    });
    // Set the hardware context when encountering different brgemm objects
    if (get_current() != value) {
      // TODO Call release_hw_context at the end when there are no calculation issues
      dnnl::ukernel::brgemm::release_hw_context();
      ((*value).brg).set_hw_context();
      get_current() = value;
    }
    // Execute brgemm
    ((*value).brg).execute(A, B, offsets, C, (*value).scratchpad.data());
  }

  // Fetch/create GemmHelper object and execute brgemm without batch size
  template <typename scalar_t_a, typename scalar_t_b, typename scalar_t_c>
  static inline void call(
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t ld_a,
      int64_t ld_b,
      int64_t ld_c,
      const float alpha,
      const float beta,
      const scalar_t_a* A,
      const scalar_t_b* B,
      scalar_t_c* C) {
    auto&& key = BrgemmKey(
        M,
        N,
        K,
        int64_t(1),
        ld_a,
        ld_b,
        ld_c,
        c10::CppTypeToScalarType<scalar_t_a>::value,
        c10::CppTypeToScalarType<scalar_t_b>::value,
        c10::CppTypeToScalarType<scalar_t_c>::value,
        alpha,
        beta);
    // Fetch/create GemmHelper object
    auto&& value = fetch_or_create(key, [&]() {
      auto&& v = std::make_shared<GemmHelper>(
          M,
          N,
          K,
          1,
          ld_a,
          ld_b,
          ld_c,
          c10::CppTypeToScalarType<scalar_t_a>::value,
          c10::CppTypeToScalarType<scalar_t_b>::value,
          c10::CppTypeToScalarType<scalar_t_c>::value,
          alpha,
          beta);
      (*v).brg.generate();
      return std::move(v);
    });
    if (get_current() != value) {
      dnnl::ukernel::brgemm::release_hw_context();
      ((*value).brg).set_hw_context();
      get_current() = value;
    }
    ((*value).brg)
        .execute(A, B, (*value).A_B_offsets, C, (*value).scratchpad.data());
  }

  static inline std::shared_ptr<GemmHelper>& get_current() {
    static thread_local std::shared_ptr<GemmHelper> current;
    return current;
  }

  static inline bool device_check(ScalarType dtype) {
    if (dtype == ScalarType::Half) {
      static bool fp16_support = dnnl::get_effective_cpu_isa() >= dnnl::cpu_isa::avx512_core_fp16;
      return fp16_support;
    } else if (dtype == ScalarType::BFloat16) {
      static bool bf16_support = dnnl::get_effective_cpu_isa() >= dnnl::cpu_isa::avx512_core;
      return bf16_support;
    } else {
      return false;
    }
  }
};

using pack_t = dnnl::ukernel::brgemm_pack_B;
struct Pack : public Kernel_Cache<PackKey, pack_t> {
  static inline void call(
      int64_t K,
      int64_t N,
      int64_t ld_in,
      int64_t ld_out,
      ScalarType dt_in,
      ScalarType dt_out,
      const void* in,
      void* out) {
    auto&& key = PackKey(K, N, ld_in, ld_out, dt_in, dt_out);
    auto&& pack = fetch_or_create(key, [&]() {
      auto&& p = std::make_shared<pack_t>(
          K, N, ld_in, ld_out, get_dnnl_dtype(dt_in), get_dnnl_dtype(dt_out));
      if (need_pack(dt_in)) {
        (*p).generate();
      }
      return std::move(p);
    });
    if (need_pack(dt_in)) {
      (*pack).execute(in, out);
    } else {
      TORCH_CHECK(false, "No need to pack");
    }
  }

    static inline bool need_pack(ScalarType dtype) {
    if (dtype == ScalarType::Half) {
      static bool fp16_pack = dnnl::get_effective_cpu_isa() >= dnnl::cpu_isa::avx512_core_amx_fp16;
      return fp16_pack;
    } else if (dtype == ScalarType::BFloat16) {
      static bool bf16_pack = dnnl::get_effective_cpu_isa() >= dnnl::cpu_isa::avx512_core_amx;
      return bf16_pack;
    } else {
      return false;
    }
  }
};

#endif

// Batch-reduce GEMM
// Operates by the following formula:
// C = alpha * SUM(A[i] x B[i]) + beta * C, i = 0 to batch size
// A Base pointer to a tensor A.
// B Base pointer to a tensor B.
// Byte offsets vector of pairs of tensors A and B offsets for
//     each batch. The number of batches must coincide with the
//     `batch_size` value passed at object construction stage.
// C Pointer to a tensor C (accumulation buffer).
// scratchpad Pointer to a scratchpad buffer.
void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t bs,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const float alpha,
    const float beta,
    const at::Half* A,
    const at::Half* B,
    const std::vector<std::pair<int64_t, int64_t>>& offsets,
    float* C);

void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t bs,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const float alpha,
    const float beta,
    const at::BFloat16* A,
    const at::BFloat16* B,
    const std::vector<std::pair<int64_t, int64_t>>& offsets,
    float* C);

// Batch-reduce GEMM
// Batch size is 1, which means it is equivalent to gemm
void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const float alpha,
    const float beta,
    const at::Half* A,
    const at::Half* B,
    float* C);

void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const float alpha,
    const float beta,
    const at::BFloat16* A,
    const at::BFloat16* B,
    float* C);

// Release brgemm hardware context
void brgemm_release();

// Pack B matrix to get better performance if needed
void pack(
    int64_t K,
    int64_t N,
    int64_t ld_in,
    int64_t ld_out,
    ScalarType dt_in,
    ScalarType dt_out,
    const void* in,
    void* out);

// Whether pack is needed in the platform.
bool need_pack(ScalarType dt_in);

} // namespace at::native::cpublas

#if AT_MKLDNN_ENABLED() && (defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC)))
namespace std {
template <>
struct hash<at::native::cpublas::BrgemmKey> {
  std::size_t operator()(const at::native::cpublas::BrgemmKey& key) const {
    std::size_t h = std::hash<bool>()(key.beta + 1);
    h = std::hash<int64_t>()(key.M) ^ (h << 1);
    h = std::hash<int64_t>()(key.N) ^ (h << 1);
    h = std::hash<int64_t>()(key.K) ^ (h << 1);
    return h;
  }
};

template <>
struct hash<at::native::cpublas::PackKey> {
  std::size_t operator()(const at::native::cpublas::PackKey& key) const {
    std::size_t h = std::hash<int64_t>()(key.K);
    h = std::hash<int64_t>()(key.N) ^ (h << 1);
    return h;
  }
};
} // namespace std
#endif
