#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/int_mm_kernel.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/Unroll.h>
#include <c10/util/irange.h>

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

#if AT_MKLDNN_ENABLED()
#include <ideep.hpp>
// Add uKernel API versioning to be compatible with different oneDNN versions
// oneDNN 3.6.x updates the ukernel APIs of brgemm and brgemm_pack_B
// brgemm_pack_B is changed to transform and the setting of brgemm beta is
// changed to set_add_C
#if (IDEEP_VERSION_MAJOR == 3 && IDEEP_VERSION_MINOR == 5)
#define ONEDNN_UKERNEL_1
#elif (IDEEP_VERSION_MAJOR >= 3 && IDEEP_VERSION_MINOR >= 6)
#define ONEDNN_UKERNEL_2
#endif
#if (                                                           \
    (defined(ONEDNN_UKERNEL_1) || defined(ONEDNN_UKERNEL_2)) && \
    (defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC))))
#define ONEDNN_UKERNEL_ENABLED
#endif
#endif // AT_MKLDNN_ENABLED()
#if defined(ONEDNN_UKERNEL_ENABLED)
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ukernel.hpp>
#endif // oneDNN BRGEMM

namespace at::native {

namespace {

template <typename T>
void int8pack_mm_kernel_(
    const Tensor& C,
    const Tensor& A,
    const Tensor& B,
    const Tensor& scales);

#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

static inline void transpose_16x16_fp32(__m512 a[16]) {
  __m512 t[16];
  c10::ForcedUnroll<16>{}([&](auto i) { t[i] = a[i]; });
  c10::ForcedUnroll<8>{}([&](auto i) {
    a[i] = (__m512)_mm512_unpacklo_pd((__m512d)t[i * 2], (__m512d)t[i * 2 + 1]);
    a[i + 8] =
        (__m512)_mm512_unpackhi_pd((__m512d)t[i * 2], (__m512d)t[i * 2 + 1]);
  });
  c10::ForcedUnroll<8>{}([&](auto i) {
    t[2 * i] = _mm512_shuffle_f32x4(a[2 * i], a[2 * i + 1], 0x44);
    t[2 * i + 1] = _mm512_shuffle_f32x4(a[2 * i], a[2 * i + 1], 0xee);
  });
  c10::ForcedUnroll<4>{}([&](auto i) {
    a[i + 0] = _mm512_shuffle_f32x4(t[4 * i + 0], t[4 * i + 2], 0x88);
    a[i + 4] = _mm512_shuffle_f32x4(t[4 * i + 0], t[4 * i + 2], 0xdd);
    a[i + 8] = _mm512_shuffle_f32x4(t[4 * i + 1], t[4 * i + 3], 0x88);
    a[i + 12] = _mm512_shuffle_f32x4(t[4 * i + 1], t[4 * i + 3], 0xdd);
  });
}

void dequant_pack_k_16(
    const int8_t* B,
    BFloat16* B_packed,
    const BFloat16* scales,
    const int ldb,
    const int n) {
  int ldb_packed = n;
  __m512 scale[16];
  __m512 vb[16];
  for (int nn = 0; nn < n / 16; nn++) {
    c10::ForcedUnroll<8>{}([&](auto i) {
      float ss = static_cast<float>(scales[i + nn * 8]);
      scale[i] = _mm512_set1_ps(ss);
      __m128i b8 = _mm_load_si128((__m128i*)(B + (i + nn * 8) * ldb));
      __m512i b32 = _mm512_cvtepi8_epi32(b8);
      vb[i] = _mm512_cvtepi32_ps(b32);
      vb[i] = _mm512_mul_ps(vb[i], scale[i]);
    });
    c10::ForcedUnroll<8>{}([&](auto i) {
      float ss = static_cast<float>(scales[i + n / 2 + nn * 8]);
      scale[i + 8] = _mm512_set1_ps(ss);
      __m128i b8 = _mm_load_si128((__m128i*)(B + (i + n / 2 + nn * 8) * ldb));
      __m512i b32 = _mm512_cvtepi8_epi32(b8);
      vb[i + 8] = _mm512_cvtepi32_ps(b32);
      vb[i + 8] = _mm512_mul_ps(vb[i + 8], scale[i + 8]);
    });
    transpose_16x16_fp32(vb);
    c10::ForcedUnroll<16>{}([&](auto i) {
      _mm256_storeu_epi16(
          (void*)(B_packed + i * ldb_packed + nn * 16),
          vec::cvtfp32_bf16(vb[i]));
    });
  }
}

template <>
void int8pack_mm_kernel_<BFloat16>(
    const Tensor& C,
    const Tensor& A,
    const Tensor& B,
    const Tensor& scales) {
  const BFloat16* A_data = A.const_data_ptr<BFloat16>();
  const int8_t* B_data = B.const_data_ptr<int8_t>();
  BFloat16* C_data = C.data_ptr<BFloat16>();
  const BFloat16* S_data = scales.const_data_ptr<BFloat16>();

  int M = A.size(0);
  int N = B.size(0);
  int K = A.size(1);
  int lda = K;
  int ldb = K;
  int ldc = N;
  const int thread_num = get_num_threads();
  const int L2_cache = 4.8 * 1024 * 1024;
  const int n_block = 32;
  int k_block = std::min(K, (L2_cache / (n_block * 3)) / 16 * 16);
  int k_tail = (K % k_block + 15) / 16 * 16;
  if (k_tail == 0)
    k_tail = k_block;
  // n=32 has a better performance than 16.
  static dnnl::ukernel::brgemm brg, brg_tail;
  static bool init = false;
  static size_t max_scratchpad_size = -1;
  if (!init) {
    brg = dnnl::ukernel::brgemm(
        M,
        32,
        k_block,
        1,
        lda,
        n_block,
        n_block,
        dnnl::memory::data_type::bf16,
        dnnl::memory::data_type::bf16,
        dnnl::memory::data_type::f32);
    brg.set_add_C(true);
    brg.finalize();
    brg.generate();
    brg_tail = dnnl::ukernel::brgemm(
        M,
        n_block,
        k_tail,
        1,
        lda,
        n_block,
        n_block,
        dnnl::memory::data_type::bf16,
        dnnl::memory::data_type::bf16,
        dnnl::memory::data_type::f32);
    brg_tail.set_post_ops(N, dnnl::memory::data_type::bf16);
    brg_tail.set_add_C(true);
    brg_tail.finalize();
    brg_tail.generate();
    max_scratchpad_size =
        std::max(max_scratchpad_size, brg.get_scratchpad_size());
    max_scratchpad_size =
        std::max(max_scratchpad_size, brg_tail.get_scratchpad_size());
    init = true;
  }
  brg.set_hw_context();
  at::parallel_for(0, N / n_block, 0, [&](int begin, int end) {
    int local_begin = begin * n_block;
    const int8_t* B_local = B_data + local_begin * ldb;
    const BFloat16* scales_local = S_data + local_begin;
    BFloat16* C_local = C_data + local_begin;

    int local_n = (end - begin) * n_block;
    if (end == N / n_block && N % n_block != 0) {
      local_n = N - local_begin;
    }
    float* tmp_local = new float[M * n_block]; // todo: aline alloc?
    BFloat16* B_pack_local = new BFloat16[k_block * n_block];
    uint8_t scratchpad_local[max_scratchpad_size];
    for (int n = 0; n < local_n; n += n_block) {
      int nn = std::min(n, local_n - 16);
      for (int k_begin = 0; k_begin < K; k_begin += k_block) {
        bool is_tail = k_begin + k_block >= K;
        int k_local = is_tail ? k_tail : k_block;
        int k_back = is_tail ? K - k_begin - k_tail : 0;
        if (k_back < 0) {
          int8_t k_tmp[16 * n_block] = {0};
          for (int ii = 0; ii < n_block; ii++)
            memcpy(
                k_tmp + 16 + k_back + nn * 16,
                B_local + (nn + ii) * ldb + k_begin,
                -k_back);
          dequant_pack_k_16(k_tmp, B_pack_local, scales_local + nn, K, n_block);
        } else {
          dequant_pack_k_16(
              B_local + nn * ldb + k_begin,
              B_pack_local,
              scales_local + nn,
              K,
              n_block);
        }
        for (int kk = 16; kk < k_local; kk += 16) {
          dequant_pack_k_16(
              B_local + nn * ldb + k_begin + kk,
              B_pack_local + kk * n_block,
              scales_local + nn,
              K,
              n_block);
        }
        if (is_tail) {
          brg_tail.execute(
              A_data + k_begin + k_back,
              B_pack_local,
              {{0, 0}},
              tmp_local,
              C_local + nn,
              scratchpad_local);
        } else {
          brg.execute(
              A_data + k_begin,
              B_pack_local,
              {{0, 0}},
              tmp_local,
              scratchpad_local);
        }
      }
    }
    delete[] tmp_local;
    delete[] B_pack_local;
  });
  dnnl::ukernel::brgemm::release_hw_context();
}

// A block : {BLOCK_M, BLOCK_K}, lda = K
// B block : {BLOCK_K, BLOCK_N}, ldb = K
// C block : {BLOCK_M, BLOCK_N}, ldc = N
//
// scales block: {BLOCK_N}
//
template <int BLOCK_M, int BLOCK_N>
inline void tinygemm_kernel(
    const BFloat16* RESTRICT A,
    const int8_t* RESTRICT B,
    const BFloat16* RESTRICT scales,
    BFloat16* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K) {
  constexpr int ROWS = BLOCK_M;
  constexpr int COLS = BLOCK_N;

  const int PREFETCH_SIZE_K = 16 * 4;

  __m512 va;
  __m512 vb[COLS];
  __m512 vc[ROWS * COLS];
  __m512 scale[COLS];

  auto load_scale = [&](int i) {
    float ss = static_cast<float>(scales[i]);
    scale[i] = _mm512_set1_ps(ss);
  };
  c10::ForcedUnroll<COLS>{}(load_scale);

  auto loadc = [&](auto i) { vc[i] = _mm512_setzero_ps(); };
  c10::ForcedUnroll<ROWS * COLS>{}(loadc);

  auto compute = [&](auto i, int k) {
    constexpr int row = i / COLS;
    constexpr int col = i % COLS;

    if constexpr (col == 0) {
      __m256i a16 = _mm256_load_si256((__m256i*)(A + row * lda + k));
      if (k + PREFETCH_SIZE_K < K) {
        _mm_prefetch(A + row * lda + k + PREFETCH_SIZE_K, _MM_HINT_T0);
      }
      vec::cvtbf16_fp32(a16, va);
    }

    if constexpr (row == 0) {
      __m128i b8 = _mm_load_si128((__m128i*)(B + col * ldb + k));
      if (k + PREFETCH_SIZE_K < K) {
        _mm_prefetch(B + col * ldb + k + PREFETCH_SIZE_K, _MM_HINT_T0);
      }
      __m512i b32 = _mm512_cvtepi8_epi32(b8);
      vb[col] = _mm512_cvtepi32_ps(b32);
      vb[col] = _mm512_mul_ps(vb[col], scale[col]);
    }

    constexpr int idx = row * COLS + col;
    vc[idx] = _mm512_fmadd_ps(va, vb[col], vc[idx]);
  };

  for (int k = 0; k < K; k += 16) {
    c10::ForcedUnroll<ROWS * COLS>{}(compute, k);
  }

  auto storec = [&](auto i) {
    constexpr int row = i / COLS;
    constexpr int col = i % COLS;
    C[row * ldc + col] = static_cast<BFloat16>(_mm512_reduce_add_ps(vc[i]));
  };
  c10::ForcedUnroll<ROWS * COLS>{}(storec);
}

#elif defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)

static inline float _mm256_reduce_add_ps(__m256& v) {
  __m256 v1 = _mm256_permute2f128_ps(v, v, 0x1);
  v = _mm256_add_ps(v, v1);
  v1 = _mm256_shuffle_ps(v, v, 0x4E);
  v = _mm256_add_ps(v, v1);
  v1 = _mm256_shuffle_ps(v, v, 0xB1);
  v = _mm256_add_ps(v, v1);
  return _mm256_cvtss_f32(v);
}

template <int BLOCK_M, int BLOCK_N>
inline void tinygemm_kernel(
    const BFloat16* RESTRICT A,
    const int8_t* RESTRICT B,
    const BFloat16* RESTRICT scales,
    BFloat16* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K) {

  constexpr int ROWS = BLOCK_M;
  constexpr int COLS = BLOCK_N;

  const int PREFETCH_SIZE_K = 16 * 4;

  __m256 va;
  __m256 vb[COLS];
  __m256 vc[ROWS * COLS];
  __m256 scale[COLS];

  auto load_scale = [&](int i) {
    float ss = static_cast<float>(scales[i]);
    scale[i] = _mm256_set1_ps(ss);
  };
  c10::ForcedUnroll<COLS>{}(load_scale);

  auto loadc = [&](auto i) {
    vc[i] = _mm256_setzero_ps();
  };
  c10::ForcedUnroll<ROWS * COLS>{}(loadc);

  auto compute = [&](auto i, int k) {
    constexpr int row = i / COLS;
    constexpr int col = i % COLS;

    if constexpr (col == 0) {
      __m128i a16 = _mm_load_si128((__m128i*)(A + row * lda + k));
      if (k + PREFETCH_SIZE_K < K) {
        _mm_prefetch(A + row * lda + k + PREFETCH_SIZE_K, _MM_HINT_T0);
      }
      vec::cvtbf16_fp32(a16, va);
    }

    if constexpr (row == 0) {
       __m128i b8 = _mm_loadu_si64((__m128i*)(B + col * ldb + k));
       if (k + PREFETCH_SIZE_K < K) {
         _mm_prefetch(B + col * ldb + k + PREFETCH_SIZE_K, _MM_HINT_T0);
       }
       __m256i b32 = _mm256_cvtepi8_epi32(b8);
       vb[col] = _mm256_cvtepi32_ps(b32);
       vb[col] = _mm256_mul_ps(vb[col], scale[col]);
     }

     constexpr int idx = row * COLS + col;
     vc[idx] = _mm256_fmadd_ps(va, vb[col], vc[idx]);
  };

  for (int k = 0; k < K; k += 8) {
    c10::ForcedUnroll<ROWS * COLS>{}(compute, k);
  }

  auto storec = [&](auto i) {
    constexpr int row = i / COLS;
    constexpr int col = i % COLS;
    C[row * ldc + col] = static_cast<BFloat16>(_mm256_reduce_add_ps(vc[i]));
  };
  c10::ForcedUnroll<ROWS * COLS>{}(storec);
}

#endif

#if !defined(C10_MOBILE) && defined(__aarch64__)
#include <arm_neon.h>

inline float reduce(float32x4_t x) {
        auto sum = vpaddq_f32(x, x);
        return vgetq_lane_f32(vpaddq_f32(sum, sum), 0);
}

inline float32x4x2_t load_as_float32x4x2(const Half* ptr) {
  float16x8_t f16_val = vld1q_f16(reinterpret_cast<const float16_t *>(ptr));
  auto val_low = vcvt_f32_f16(vget_low_f16(f16_val));
  auto val_high = vcvt_f32_f16(vget_high_f16(f16_val));
  return {val_low, val_high};
}

inline float32x4_t load_as_float32x4(const Half* ptr) {
    return vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(ptr)));
}

inline float32x4x2_t load_as_float32x4x2(const BFloat16* ptr) {
  int32x4_t shift = vdupq_n_s32(16);
  uint16x8_t u16_val = vld1q_u16(reinterpret_cast<const uint16_t *>(ptr));
  uint32x4_t int_low = vmovl_u16(vget_low_u16(u16_val));
  uint32x4_t int_high = vmovl_u16(vget_high_u16(u16_val));
  return {vreinterpretq_f32_u32(vshlq_u32(int_low, shift)), vreinterpretq_f32_u32(vshlq_u32(int_high, shift))};
}

inline float32x4_t load_as_float32x4(const BFloat16* ptr) {
  int32x4_t shift = vdupq_n_s32(16);
  uint32x4_t as_int = vmovl_u16(vld1_u16(reinterpret_cast<const uint16_t *>(ptr)));
  return vreinterpretq_f32_u32(vshlq_u32(as_int, shift));
}

inline float32x4_t load_as_float32x4(const float* ptr) {
  return vld1q_f32(ptr);
}

inline float32x4x2_t load_as_float32x4x2(const float* ptr) {
  return {vld1q_f32(ptr), vld1q_f32(ptr + 4)};
}

template <int BLOCK_M, int BLOCK_N, typename T>
inline void tinygemm_kernel_(
    const T* RESTRICT A,
    const int8_t* RESTRICT B,
    const T* RESTRICT scales,
    T* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K) {

  for (const auto m : c10::irange(BLOCK_M)) {
    float32x4_t c_val[BLOCK_N];
    c10::ForcedUnroll<BLOCK_N>{}([&](auto i) {
        c_val[i] = vdupq_n_f32(0.0);
    });
    for (int k = 0; k < K; k += 8) {
      auto a_val = load_as_float32x4x2(A + m * lda + k);
      c10::ForcedUnroll<BLOCK_N>{}([&](auto i) {
        int16x8_t b_val = vmovl_s8(vld1_s8(B + i * ldb + k));
        auto b_val_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b_val)));
        auto b_val_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b_val)));
        c_val[i] = vfmaq_f32(c_val[i], a_val.val[1], b_val_high);
        c_val[i] = vfmaq_f32(c_val[i], a_val.val[0], b_val_low);
      });
    }

#if __OPTIMIZE__
    float32x4_t scale_val = load_as_float32x4(scales);
    c10::ForcedUnroll<BLOCK_N>{}([&](auto i) {
      C[m * ldc + i] = reduce(c_val[i]) * vgetq_lane_f32(scale_val, i);
    });
#else
    // Workaround GCCs inability to infer lane index at compile time
    // See https://github.com/pytorch/pytorch/issues/126283
    c10::ForcedUnroll<BLOCK_N>{}([&](auto i) {
      C[m * ldc + i] = reduce(c_val[i]) * float(scales[i]);
    });
#endif
  }
}

template <int BLOCK_M, int BLOCK_N>
inline void tinygemm_kernel(
    const Half* RESTRICT A,
    const int8_t* RESTRICT B,
    const Half* RESTRICT scales,
    Half* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K) {
  tinygemm_kernel_<BLOCK_M, BLOCK_N>(A, B, scales, C, lda, ldb, ldc, K);
}

template <int BLOCK_M, int BLOCK_N>
inline void tinygemm_kernel(
    const BFloat16* RESTRICT A,
    const int8_t* RESTRICT B,
    const BFloat16* RESTRICT scales,
    BFloat16* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K) {
  tinygemm_kernel_<BLOCK_M, BLOCK_N>(A, B, scales, C, lda, ldb, ldc, K);
}

template <int BLOCK_M, int BLOCK_N>
inline void tinygemm_kernel(
    const float* RESTRICT A,
    const int8_t* RESTRICT B,
    const float* RESTRICT scales,
    float* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K) {
  tinygemm_kernel_<BLOCK_M, BLOCK_N>(A, B, scales, C, lda, ldb, ldc, K);
}
#endif

// non-vectorized version
template <int BLOCK_M, int BLOCK_N, typename T>
inline void tinygemm_kernel(
    const T* RESTRICT A,
    const int8_t* RESTRICT B,
    const T* RESTRICT scales,
    T* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K) {

  for (const auto m : c10::irange(BLOCK_M)) {
    for (const auto n : c10::irange(BLOCK_N)) {
      float c_val = 0;
      float scale_val = static_cast<float>(scales[n]);
      for (const auto k : c10::irange(K)) {
        float a_val = static_cast<float>(A[m * lda + k]);
        float b_val = static_cast<float>(B[n * ldb + k]);
        c_val += a_val * (b_val * scale_val);
      }
      C[m * ldc + n] = c_val;
    }
  }
}

#define LAUNCH_TINYGEMM_KERNEL(MB_SIZE, NB_SIZE)                 \
  tinygemm_kernel<MB_SIZE, NB_SIZE>(                             \
      A_ptr, B_ptr, S_ptr, C_ptr,                                \
      K, K, N, K);

#define LAUNCH_TINYGEMM_NB_SIZE(MB_SIZE)                         \
  switch (nb_size) {                                             \
    case 1:                                                      \
      LAUNCH_TINYGEMM_KERNEL(MB_SIZE, 1);                        \
      break;                                                     \
    case 2:                                                      \
      LAUNCH_TINYGEMM_KERNEL(MB_SIZE, 2);                        \
      break;                                                     \
    case 3:                                                      \
      LAUNCH_TINYGEMM_KERNEL(MB_SIZE, 3);                        \
      break;                                                     \
    case 4:                                                      \
      LAUNCH_TINYGEMM_KERNEL(MB_SIZE, 4);                        \
      break;                                                     \
    default:                                                     \
      TORCH_CHECK(false, "Unsupported n block size: ", nb_size); \
      break;                                                     \
  }

template<typename T>
void int8pack_mm_kernel_(
    const Tensor& C,
    const Tensor& A,
    const Tensor& B,
    const Tensor& scales) {

  const auto* A_data = A.const_data_ptr<T>();
  const auto* B_data = B.const_data_ptr<int8_t>();
  auto* C_data = C.data_ptr<T>();
  const auto* S_data = scales.const_data_ptr<T>();

  int M = A.size(0);
  int N = B.size(0);
  int K = A.size(1);

  constexpr int BLOCK_M = 4;
  constexpr int BLOCK_N = 4;

  const int MB = (M + BLOCK_M - 1) / BLOCK_M;
  const int NB = (N + BLOCK_N - 1) / BLOCK_N;

  at::parallel_for(0, MB * NB, 0, [&](int begin, int end) {
    int mb{0}, nb{0};
    data_index_init(begin, mb, MB, nb, NB);

    for (const auto i : c10::irange(begin, end)) {
      (void)i;

      int mb_start = mb * BLOCK_M;
      int mb_size = std::min(BLOCK_M, M - mb_start);
      int nb_start = nb * BLOCK_N;
      int nb_size = std::min(BLOCK_N, N - nb_start);

      const auto* A_ptr = A_data + mb_start * K;
      const auto* B_ptr = B_data + nb_start * K;
      const auto* S_ptr = S_data + nb_start;
      auto* C_ptr = C_data + mb_start * N + nb_start;

      switch (mb_size) {
        case 1:
          LAUNCH_TINYGEMM_NB_SIZE(1);
          break;
        case 2:
          LAUNCH_TINYGEMM_NB_SIZE(2);
          break;
        case 3:
          LAUNCH_TINYGEMM_NB_SIZE(3);
          break;
        case 4:
          LAUNCH_TINYGEMM_NB_SIZE(4);
          break;
        default:
          TORCH_CHECK(false, "Unsupported m block size: ", mb_size);
      }

      // move to the next index
      data_index_step(mb, MB, nb, NB);
    }
  });
}

void int8pack_mm_kernel(
    const Tensor& C,
    const Tensor& A,
    const Tensor& B,
    const Tensor& scales) {
  if (C.dtype() == kHalf) {
    int8pack_mm_kernel_<Half>(C, A, B, scales);
  } else if (C.dtype() == kBFloat16) {
    int8pack_mm_kernel_<BFloat16>(C, A, B, scales);
  } else {
    int8pack_mm_kernel_<float>(C, A, B, scales);
  }
}

} // anonymous namespace

ALSO_REGISTER_AVX512_DISPATCH(int8pack_mm_stub, &int8pack_mm_kernel)

} // at::native
