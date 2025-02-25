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

#if (defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2)) && !defined(_MSC_VER)

// A block : {BLOCK_M, BLOCK_K}, lda = K
// B block : {BLOCK_K, BLOCK_N}, ldb = K
// C block : {BLOCK_M, BLOCK_N}, ldc = N
//
// scales block: {BLOCK_N}
//
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

  constexpr int ROWS = BLOCK_M;
  constexpr int COLS = BLOCK_N;

  const int PREFETCH_SIZE_K = 16 * 4;

  using Vectorized = at::vec::Vectorized<float>;
  using VectorizedIn = at::vec::Vectorized<T>;
  constexpr auto VLEN = Vectorized::size();

  Vectorized va;
  at::vec::VectorizedN<float, COLS> vb;
  at::vec::VectorizedN<float, ROWS * COLS> vc;
  at::vec::VectorizedN<float, COLS> scale;

  auto load_scale = [&](int i) {
    scale[i] = Vectorized(static_cast<float>(scales[i]));
  };
  c10::ForcedUnroll<COLS>{}(load_scale);

  auto loadc = [&](auto i) {
    vc[i] = Vectorized(0.0f);
  };
  c10::ForcedUnroll<ROWS * COLS>{}(loadc);

  auto compute = [&](auto i, int k) {
    constexpr int row = i / COLS;
    constexpr int col = i % COLS;

    if constexpr (col == 0) {
      auto a16 = VectorizedIn::loadu(A + row * lda + k, VLEN);
      if (k + PREFETCH_SIZE_K < K) {
        _mm_prefetch(A + row * lda + k + PREFETCH_SIZE_K, _MM_HINT_T0);
      }
      va = at::vec::convert<float>(a16);
    }

    if constexpr (row == 0) {
      auto b8 = at::vec::Vectorized<int8_t>::loadu(B + col * ldb + k, VLEN);
      if (k + PREFETCH_SIZE_K < K) {
        _mm_prefetch(B + col * ldb + k + PREFETCH_SIZE_K, _MM_HINT_T0);
      }
      vb[col] = at::vec::convert<float>(b8);
      vb[col] = vb[col] * scale[col];
    }

    constexpr int idx = row * COLS + col;
    vc[idx] = at::vec::fmadd(va, vb[col], vc[idx]);
  };

  for (int k = 0; k < K; k += VLEN) {
      c10::ForcedUnroll<ROWS * COLS>{}(compute, k);
  }

  auto storec = [&](auto i) {
    constexpr int row = i / COLS;
    constexpr int col = i % COLS;
    C[row * ldc + col] = static_cast<T>(vc[i].reduce_add());
  };
  c10::ForcedUnroll<ROWS * COLS>{}(storec);
}

#else

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

#define LAUNCH_TINYGEMM_KERNEL(MB_SIZE, NB_SIZE)                 \
  tinygemm_kernel<MB_SIZE, NB_SIZE>(                             \
      A_ptr, B_ptr, S_ptr, C_ptr,                                \
      lda, K, N, K);

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
  int lda = A.stride(0);
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

      const auto* A_ptr = A_data + mb_start * lda;
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
