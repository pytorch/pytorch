#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/int_mm_kernel.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>
#include <c10/util/Unroll.h>

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

namespace at::native {

namespace {

#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

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

  auto loadc = [&](auto i) {
    vc[i] = _mm512_setzero_ps();
  };
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

#else

// non-vectorized version
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

#endif

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

void int8pack_mm_kernel(
    const Tensor& C,
    const Tensor& A,
    const Tensor& B,
    const Tensor& scales) {

  const auto* A_data = A.data_ptr<BFloat16>();
  const auto* B_data = B.data_ptr<int8_t>();
  auto* C_data = C.data_ptr<BFloat16>();
  const auto* S_data = scales.data_ptr<BFloat16>();

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

} // anonymous namespace

ALSO_REGISTER_AVX512_DISPATCH(int8pack_mm_stub, &int8pack_mm_kernel);

} // at::native
