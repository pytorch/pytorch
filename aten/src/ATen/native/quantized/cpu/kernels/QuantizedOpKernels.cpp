#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TopKImpl.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UpSample.h>
#include <ATen/native/cpu/IndexKernelUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/quantized/FakeQuantAffine.h>
#include <ATen/native/quantized/IndexKernel.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>
#include <c10/util/Unroll.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/empty.h>
#endif

#include <cmath>
#ifdef USE_FBGEMM
#include <fbgemm/QuantUtils.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__ARM_NEON__) || defined(__aarch64__)
#include <ATen/quantized/Quantizer.h>
#include <arm_neon.h>
#endif


// NOLINTBEGIN(*-c-arrays)
namespace at::native {
namespace {

void check_tensor_memory_format(const Tensor& ref, const Tensor& other) {
  TORCH_CHECK(
      ref.is_contiguous(ref.suggest_memory_format()),
      "Quantized tensor should be contiguous");
  TORCH_CHECK(
      other.is_contiguous(ref.suggest_memory_format()),
      "Float tensor should be contiguous "
      "in same memory format as quantized tensor");
}

// ****************** HEY YOU! YES YOU! Read this! ********************
//
// Please read the README.md in this directory before editing this file

template <bool ReLUFused = false>
Tensor qcat_nhwc_kernel(
    const MaterializedITensorListRef& qxs,
    int64_t dim,
    double scale,
    int64_t zero_point) {
  const at::Tensor& qx0 = qxs[0];
  int64_t C_out = 0;
  std::vector<int64_t> Cs_in;
  // Prefix sum of input channels for fast indexing
  std::vector<int64_t> Cs_sum;
  std::vector<double> scales;
  std::vector<int64_t> zero_pts;
  std::vector<void*> data_ptrs;
  std::vector<bool> is_fast_path;

  for (const at::Tensor& qx : qxs) {
    TORCH_CHECK(
        qx.dim() == qx0.dim(),
        "Tensors must have the same number of dimensions: got ",
        qx.dim(),
        " and ",
        qx0.dim());
#define CHECK_DIM(d)                                            \
  TORCH_CHECK(                                                  \
      qx.size(d) == qx0.size(d),                                \
      "Sizes of tensors must match expect in dimension 1. Got", \
      qx.size(d),                                               \
      " and ",                                                  \
      qx0.size(d));
    CHECK_DIM(0);
    CHECK_DIM(2);
    CHECK_DIM(3);
    TORCH_CHECK(
        qx.scalar_type() == qx0.scalar_type(),
        "Expected object of scalar type ",
        toString(qx0.scalar_type()),
        " but got scalar type ",
        toString(qx.scalar_type()));
    Cs_in.push_back(qx.size(1));
    Cs_sum.push_back(C_out);
    C_out += qx.size(1);
    scales.push_back(qx.q_scale());
    zero_pts.push_back(qx.q_zero_point());
    data_ptrs.push_back(qx.data_ptr());
    is_fast_path.push_back(
        qx.q_scale() == scale &&
        qx.q_zero_point() == zero_point);
  }

  const int64_t N = qx0.size(0);
  const int64_t H = qx0.size(2);
  const int64_t W = qx0.size(3);
  float inv_scale = static_cast<float>(1.0 / scale);

  auto output = at::_empty_affine_quantized(
      {N, C_out, H, W},
      qx0.options().memory_format(MemoryFormat::ChannelsLast),
      scale,
      zero_point,
      std::nullopt);

  // N, H, and W are explicitly captured here because there's a bug in GCC5
  // and clang5 which causes an internal compiler error if they're not
  AT_DISPATCH_QINT_TYPES(output.scalar_type(), "qcat_nhwc", [&, N, H, W]() {
    using Vec = Vectorized<scalar_t>;
    at::parallel_for(0, N * H * W, 0, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        // loop over input tensors
        for (const auto tidx : c10::irange(Cs_in.size())) {
          scalar_t::underlying* optr =
              reinterpret_cast<scalar_t::underlying*>(output.data_ptr()) +
              i * C_out + Cs_sum[tidx];

          auto curr_C = Cs_in[tidx];
          float curr_scale = scales[tidx];
          int64_t curr_zero_pt = zero_pts[tidx];

          scalar_t::underlying* iptr =
              reinterpret_cast<scalar_t::underlying*>(data_ptrs[tidx]) +
              i * curr_C;

          if (is_fast_path[tidx] && !ReLUFused) {
            std::memcpy(optr, iptr, curr_C * sizeof(typename scalar_t::underlying));
            continue;
          }

          constexpr auto VLEN = Vec::size();
          int64_t c = 0;

          // Vectorized loop
          if (c + VLEN <= curr_C) {
            auto curr_scale_vec = Vectorized<float>(curr_scale);
            auto curr_zero_pt_vec = Vectorized<float>(curr_zero_pt);
            auto scale_neg_zp_premul = curr_scale_vec * curr_zero_pt_vec.neg();
            for (; c + VLEN <= curr_C; c += VLEN) {
              auto inp_vec = Vec::loadu(iptr + c);
              auto float_values = inp_vec.dequantize(
                  curr_scale_vec, curr_zero_pt_vec, scale_neg_zp_premul);
              Vec::float_vec_return_type retvals;
              for (int i = 0; i < Vec::float_num_vecs(); ++i) {
                if constexpr (ReLUFused) {
                  retvals[i] =
                      vec::maximum(float_values[i], Vectorized<float>(0.0f));
                } else {
                  retvals[i] = float_values[i];
                }
              }
              auto quantized =
                  Vec::quantize(retvals, scale, zero_point, inv_scale);
              quantized.store(optr + c);
            }
          }

          // Vectorized loop for channel between 8 and 32 (avx2)
          constexpr auto kVLEN = Vectorized<float>::size();
          int64_t elem_size = curr_C - c;
          if ((VLEN == 4 * kVLEN) && elem_size >= kVLEN) {
            auto curr_scale_vec = Vectorized<float>(curr_scale);
            auto curr_zero_pt_vec = Vectorized<float>(curr_zero_pt);
            auto scale_neg_zp_premul = curr_scale_vec * curr_zero_pt_vec.neg();
            int64_t vec_num = elem_size / kVLEN;
            std::array<typename scalar_t::underlying, VLEN> buf_in{};
            memcpy(buf_in.data(), iptr + c, vec_num * kVLEN);
            auto inp_vec = Vec::loadu(buf_in.data());
            auto float_values = inp_vec.dequantize(
                curr_scale_vec, curr_zero_pt_vec, scale_neg_zp_premul);
            Vec::float_vec_return_type retvals;
            for (int i = 0; i < vec_num; ++i) {
              if constexpr (ReLUFused) {
                retvals[i] =
                    vec::maximum(float_values[i], Vectorized<float>(0.0f));
              } else {
                retvals[i] = float_values[i];
              }
            }
            auto quantized =
                Vec::quantize(retvals, scale, zero_point, inv_scale);
            quantized.store(optr + c, vec_num * kVLEN);
            c += vec_num * kVLEN;
          }

          // Scalar loop
          for (; c < curr_C; ++c) {
            auto float_val = at::native::dequantize_val(
                curr_scale,
                curr_zero_pt,
                reinterpret_cast<scalar_t*>(iptr)[c]);
            if constexpr (ReLUFused) {
              float_val = std::max(0.0f, float_val);
            }
            optr[c] = at::native::quantize_val<scalar_t>(
                          scale, zero_point, float_val)
                          .val_;
          } // for c
        } // for tidx
      } // for i
    });
  });

  return output;
}

// horizontal sum over a range of uint8_t
int64_t hsum(const uint8_t* A, int len) {
  int64_t row_sum = 0;
  int i = 0;

#ifdef CPU_CAPABILITY_AVX2
  __m256i sum_v = _mm256_setzero_si256();
  __m256i one_epi16_v = _mm256_set1_epi16(1);
  __m256i one_epi8_v = _mm256_set1_epi8(1);
  // vectorized
  for (; i < len / 32 * 32; i += 32) {
    __m256i src_v = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
    sum_v = _mm256_add_epi32(
      sum_v,
      _mm256_madd_epi16(
        // first argument is unsigned, second is signed
        _mm256_maddubs_epi16(src_v, one_epi8_v),
      one_epi16_v)
    );
  }

  alignas(64) int32_t temp[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_v);
  for (const auto k : c10::irange(8)) {
    row_sum += temp[k];
  }
#elif defined(CPU_CAPABILITY_AVX512)
  __m512i sum_v = _mm512_setzero_si512();
  __m512i one_epi16_v = _mm512_set1_epi16(1);
  __m512i one_epi8_v = _mm512_set1_epi8(1);
  // vectorized
  for (; i < len / 64 * 64; i += 64) {
    __m512i src_v = _mm512_loadu_si512(reinterpret_cast<__m512i const*>(A + i));
    sum_v = _mm512_add_epi32(
      sum_v,
      _mm512_madd_epi16(
        // first argument is unsigned, second is signed
        _mm512_maddubs_epi16(src_v, one_epi8_v),
      one_epi16_v)
    );
  }

  alignas(64) int32_t temp[16];
  _mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_v);
  for (const auto k : c10::irange(16)) {
    row_sum += temp[k];
  }
#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

  // scalar
  for (; i < len; ++i) {
    row_sum += A[i];
  }

  return row_sum;
}

// horizontal sum over a range of int8_t
int64_t hsum(const int8_t* A, int len) {
  int64_t row_sum = 0;
  int i = 0;

#ifdef CPU_CAPABILITY_AVX2
  __m256i sum_v = _mm256_setzero_si256();
  __m256i one_epi16_v = _mm256_set1_epi16(1);
  __m256i one_epi8_v = _mm256_set1_epi8(1);
  // vectorized
  for (; i < len / 32 * 32; i += 32) {
    __m256i src_v = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
    sum_v = _mm256_add_epi32(
      sum_v,
      _mm256_madd_epi16(
        // first argument is unsigned, second is signed
        _mm256_maddubs_epi16(one_epi8_v, src_v),
      one_epi16_v)
    );
  }

  alignas(64) int32_t temp[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_v);
  for (const auto k : c10::irange(8)) {
    row_sum += temp[k];
  }
#elif defined(CPU_CAPABILITY_AVX512)
  __m512i sum_v = _mm512_setzero_si512();
  __m512i one_epi16_v = _mm512_set1_epi16(1);
  __m512i one_epi8_v = _mm512_set1_epi8(1);
  // vectorized
  for (; i < len / 64 * 64; i += 64) {
    __m512i src_v = _mm512_loadu_si512(reinterpret_cast<__m512i const*>(A + i));
    sum_v = _mm512_add_epi32(
      sum_v,
      _mm512_madd_epi16(
        // first argument is unsigned, second is signed
        _mm512_maddubs_epi16(one_epi8_v, src_v),
      one_epi16_v)
    );
  }

  alignas(64) int32_t temp[16];
  _mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_v);
  for (const auto k : c10::irange(16)) {
    row_sum += temp[k];
  }
#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

  // scalar
  for (; i < len; ++i) {
    row_sum += A[i];
  }

  return row_sum;
}

// horizontal sum over a range of int32_t
int64_t hsum(const int32_t* A, int len) {
  int64_t row_sum = 0;
  int i = 0;

#ifdef CPU_CAPABILITY_AVX2
  __m256i sum_epi64 = _mm256_setzero_si256();
  // vectorized
  for (; i < len / 8 * 8; i += 8) {
    __m256i src_epi32 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
    // widen
    __m128i src_lo_epi32 = _mm256_castsi256_si128(src_epi32);
    __m128i src_hi_epi32 = _mm256_extracti128_si256(src_epi32, 1);
    __m256i src_lo_epi64 = _mm256_cvtepi32_epi64(src_lo_epi32);
    __m256i src_hi_epi64 = _mm256_cvtepi32_epi64(src_hi_epi32);
    // add
    sum_epi64 = _mm256_add_epi64(sum_epi64, src_lo_epi64);
    sum_epi64 = _mm256_add_epi64(sum_epi64, src_hi_epi64);
  }

  alignas(64) int64_t temp[4];
  _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_epi64);
  for (const auto k : c10::irange(4)) {
    row_sum += temp[k];
  }
#elif defined(CPU_CAPABILITY_AVX512)
  __m512i sum_epi64 = _mm512_setzero_si512();
  // vectorized
  for (; i < len / 16 * 16; i += 16) {
    __m512i src_epi32 = _mm512_loadu_si512(reinterpret_cast<__m512i const*>(A + i));
    // widen
    __m256i src_lo_epi32 = _mm512_castsi512_si256(src_epi32);
    __m256i src_hi_epi32 = _mm512_extracti32x8_epi32(src_epi32, 1);
    __m512i src_lo_epi64 = _mm512_cvtepi32_epi64(src_lo_epi32);
    __m512i src_hi_epi64 = _mm512_cvtepi32_epi64(src_hi_epi32);
    // add
    sum_epi64 = _mm512_add_epi64(sum_epi64, src_lo_epi64);
    sum_epi64 = _mm512_add_epi64(sum_epi64, src_hi_epi64);
  }

  alignas(64) int64_t temp[8];
  _mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_epi64);
  for (const auto k : c10::irange(8)) {
    row_sum += temp[k];
  }
#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

  // scalar
  for (; i < len; ++i) {
    row_sum += A[i];
  }

  return row_sum;
}

// horizontal sum of squares over a range of uint8_t
int64_t hsum_sq(const uint8_t* A, int len) {
  int64_t row_sum = 0;
  int i = 0;

#ifdef CPU_CAPABILITY_AVX2
  // vectorized
  __m256i sum_v_epu32 = _mm256_setzero_si256();
  alignas(64) int32_t temp[8];
  int overflow_threshold = 262144; // 2147483647(max of int32)/(256*256)*8 = 262144
  int loop = len / overflow_threshold + 1;
  for(int j=0; j<=loop; j++){
    for (; ((i < overflow_threshold * j) && (i < len / 16 * 16)); i += 16) {
      // (i15, ..., i0)
      __m128i src_epu8 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(A + i));
      __m256i src_epu16 = _mm256_cvtepu8_epi16(src_epu8);
      // (i15 ^ 2, ..., i0 ^ 2)
      __m256i sq_epu16 = _mm256_mullo_epi16(src_epu16, src_epu16);
      // (i7 ^ 2, ..., i0 ^ 2)
      __m128i sq_lo_epu16 = _mm256_castsi256_si128(sq_epu16);
      // (i15 ^ 2, ..., i8 ^ 2)
      __m128i sq_hi_epu16 = _mm256_extractf128_si256(sq_epu16, 1);
      // widen to epu32
      __m256i sq_lo_epu32 = _mm256_cvtepu16_epi32(sq_lo_epu16);
      __m256i sq_hi_epu32 = _mm256_cvtepu16_epi32(sq_hi_epu16);
      // add to running sum
      sum_v_epu32 = _mm256_add_epi32(sum_v_epu32, sq_lo_epu32);
      sum_v_epu32 = _mm256_add_epi32(sum_v_epu32, sq_hi_epu32);
    }
    _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_v_epu32);
    for (const auto k : c10::irange(8)) {
      row_sum += temp[k];
    }
    sum_v_epu32 = _mm256_setzero_si256();
  }
#elif defined(CPU_CAPABILITY_AVX512)
  __m512i sum_v_epu32 = _mm512_setzero_si512();
  alignas(64) int32_t temp[16];
  int overflow_threshold = 262144; // 2147483647(max of int32)/(512*512)*8 = 262144
  int loop = len / overflow_threshold + 1;
  for(int j=0; j<=loop; j++){
    for (; ((i < overflow_threshold * j) && (i < len / 32 * 32)); i += 32) {
      // (i31, ..., i0)
      __m256i src_epu8 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
      __m512i src_epu16 = _mm512_cvtepu8_epi16(src_epu8);
      // (i31 ^ 2, ..., i0 ^ 2)
      __m512i sq_epu16 = _mm512_mullo_epi16(src_epu16, src_epu16);
      // (i15 ^ 2, ..., i0 ^ 2)
      __m256i sq_lo_epu16 = _mm512_castsi512_si256(sq_epu16);
      // (i31 ^ 2, ..., i16 ^ 2)
      __m256i sq_hi_epu16 = _mm512_extracti32x8_epi32(sq_epu16, 1);
      // widen to epu32
      __m512i sq_lo_epu32 = _mm512_cvtepu16_epi32(sq_lo_epu16);
      __m512i sq_hi_epu32 = _mm512_cvtepu16_epi32(sq_hi_epu16);
      // add to running sum
      sum_v_epu32 = _mm512_add_epi32(sum_v_epu32, sq_lo_epu32);
      sum_v_epu32 = _mm512_add_epi32(sum_v_epu32, sq_hi_epu32);
    }
    _mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_v_epu32);
    for (const auto k : c10::irange(16)) {
      row_sum += temp[k];
    }
    sum_v_epu32 = _mm512_setzero_si512();
  }
#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

  // scalar
  for (; i < len; ++i) {
    row_sum += A[i] * A[i];
  }

  return row_sum;
}

// horizontal sum of squares over a range of int8_t
int64_t hsum_sq(const int8_t* A, int len) {
  int64_t row_sum = 0;
  int i = 0;

#ifdef CPU_CAPABILITY_AVX2
  // vectorized
  __m256i sum_v_epi32 = _mm256_setzero_si256();
  alignas(64) int32_t temp[8];

  int overflow_threshold = 1048576; //2147483647/(128*128)*8 = 1048576
  int loop = len / overflow_threshold + 1;

  for(int j=0; j<=loop; j++){
    for (; ((i < overflow_threshold * j) && (i < len / 16 * 16)); i += 16) {
      // (i15, ..., i0)
      __m128i src_epi8 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(A + i));
      __m256i src_epi16 = _mm256_cvtepi8_epi16(src_epi8);
      // (i15 ^ 2, ..., i0 ^ 2)
      __m256i sq_epi16 = _mm256_mullo_epi16(src_epi16, src_epi16);
      // (i7 ^ 2, ..., i0 ^ 2)
      __m128i sq_lo_epi16 = _mm256_castsi256_si128(sq_epi16);
      // (i15 ^ 2, ..., i8 ^ 2)
      __m128i sq_hi_epi16 = _mm256_extractf128_si256(sq_epi16, 1);
      // widen to epi32
      __m256i sq_lo_epi32 = _mm256_cvtepi16_epi32(sq_lo_epi16);
      __m256i sq_hi_epi32 = _mm256_cvtepi16_epi32(sq_hi_epi16);
      // add to running sum
      sum_v_epi32 = _mm256_add_epi32(sum_v_epi32, sq_lo_epi32);
      sum_v_epi32 = _mm256_add_epi32(sum_v_epi32, sq_hi_epi32);
    }
    _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_v_epi32);

    for (const auto k : c10::irange(8)) {
      row_sum += temp[k];
    }
    sum_v_epi32 = _mm256_setzero_si256();
  }
#elif defined(CPU_CAPABILITY_AVX512)
  // vectorized
  __m512i sum_v_epi32 = _mm512_setzero_si512();
  alignas(64) int32_t temp[16];

  int overflow_threshold = 1048576; //2147483647/(256*256)*8 = 1048576
  int loop = len / overflow_threshold + 1;

  for(int j=0; j<=loop; j++){
    for (; ((i < overflow_threshold * j) && (i < len / 32 * 32)); i += 32) {
      // (i31, ..., i0)
      __m256i src_epi8 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
      __m512i src_epi16 = _mm512_cvtepi8_epi16(src_epi8);
      // (i31 ^ 2, ..., i0 ^ 2)
      __m512i sq_epi16 = _mm512_mullo_epi16(src_epi16, src_epi16);
      // (i15 ^ 2, ..., i0 ^ 2)
      __m256i sq_lo_epi16 = _mm512_castsi512_si256(sq_epi16);
      // (i31 ^ 2, ..., i16 ^ 2)
      __m256i sq_hi_epi16 = _mm512_extracti32x8_epi32(sq_epi16, 1);
      // widen to epi32
      __m512i sq_lo_epi32 = _mm512_cvtepi16_epi32(sq_lo_epi16);
      __m512i sq_hi_epi32 = _mm512_cvtepi16_epi32(sq_hi_epi16);
      // add to running sum
      sum_v_epi32 = _mm512_add_epi32(sum_v_epi32, sq_lo_epi32);
      sum_v_epi32 = _mm512_add_epi32(sum_v_epi32, sq_hi_epi32);
    }
    _mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_v_epi32);

    for (const auto k : c10::irange(16)) {
      row_sum += temp[k];
    }
    sum_v_epi32 = _mm512_setzero_si512();
  }
#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

  // scalar
  for (; i < len; ++i) {
    row_sum += A[i] * A[i];
  }

  return row_sum;
}

// horizontal sum os squares over a range of int32_t
// floats throughout are necessary to prevent overflow
float hsum_sq(const int32_t* A, int len) {
  float row_sum = 0;
  int i = 0;

#ifdef CPU_CAPABILITY_AVX2
  __m256 sum_ps = _mm256_setzero_ps();
  // vectorized
  for (; i < len / 8 * 8; i += 8) {
    __m256i src_epi32 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
    __m256 src_ps = _mm256_cvtepi32_ps(src_epi32);
    sum_ps = _mm256_add_ps(sum_ps, _mm256_mul_ps(src_ps, src_ps));
  }

  alignas(64) float temp[8];
  _mm256_store_ps(temp, sum_ps);
  for (const auto k : c10::irange(8)) {
    row_sum += temp[k];
  }
#elif defined(CPU_CAPABILITY_AVX512)
  __m512 sum_ps = _mm512_setzero_ps();
  // vectorized
  for (; i < len / 16 * 16; i += 16) {
    __m512i src_epi32 = _mm512_loadu_si512(reinterpret_cast<__m512i const*>(A + i));
    __m512 src_ps = _mm512_cvtepi32_ps(src_epi32);
    sum_ps = _mm512_add_ps(sum_ps, _mm512_mul_ps(src_ps, src_ps));
  }

  alignas(64) float temp[16];
  _mm512_store_ps(temp, sum_ps);
  for (const auto k : c10::irange(16)) {
    row_sum += temp[k];
  }
#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

  // scalar
  for (; i < len; ++i) {
    int64_t cur = static_cast<int64_t>(A[i]);
    row_sum += (float)cur * (float)cur;
  }

  return row_sum;
}

void qrelu_kernel(const Tensor& qx, Tensor& qy) {
  const auto zero_point = qx.q_zero_point();
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu", [&]() {
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point(),
        std::nullopt);
    using Vec = Vectorized<scalar_t>;
    auto zero_point_vec = Vec(scalar_t(zero_point));
    auto iter = TensorIterator::unary_op(qy, qx);
    cpu_kernel_vec(
        iter,
        [&](scalar_t value) -> scalar_t {
          return scalar_t(std::max<underlying_t>(value.val_, zero_point));
        },
        [&](Vec value) -> Vec { return value.relu(zero_point_vec); });
  });
}

void leaky_qrelu_out_kernel(Tensor& out, const Tensor& qx,
                                   const Scalar& negval_) {
  int64_t i_zp = qx.q_zero_point();
  float i_scale = static_cast<float>(qx.q_scale());

  int64_t o_zp = out.q_zero_point();
  float o_scale = static_cast<float>(out.q_scale());
  float o_inv_scale = 1.0f / o_scale;

  float negval = negval_.to<float>();

  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "leaky_qrelu", [&] {
    using Vec = Vectorized<float>;  // Naive implementation uses dequant/quant loop.
    using qVec = Vectorized<scalar_t>;
    Vec zero_vec = Vec(0.0f);
    Vec one_vec = Vec(1.0f);

    Vec i_scale_vec = Vec(i_scale);
    Vec i_zp_vec = Vec(i_zp);
    Vec i_scale_zp_neg_premul_vec = i_scale_vec * i_zp_vec.neg();

    Vec negval_vec = Vec(negval);

    auto iter = TensorIterator::unary_op(out, qx);

    cpu_kernel_vec(
        iter,
        [&](scalar_t value_qx) -> scalar_t {
          auto value_dx = at::native::dequantize_val(i_scale, i_zp, value_qx);
          auto value_dy = value_dx > 0 ? value_dx : value_dx * negval;
          return at::native::quantize_val<scalar_t>(o_scale, o_zp, value_dy);
        },
        [&](qVec qx_vec) -> qVec {
          /* Vectorized implementation creates a multiplicand vector, which has
           * "alpha" for all negative dx values and ones-vector for all
           * positive values of dx. The multiplicand then is multiplied by the
           * input.
           */
          auto dx_vec_vec = qx_vec.dequantize(i_scale_vec, i_zp_vec,
                                              i_scale_zp_neg_premul_vec);
          for (auto & dx_vec : dx_vec_vec) {
            const auto multiplicand = Vec::blendv(negval_vec, one_vec,
                                                  dx_vec > zero_vec);
            dx_vec *= multiplicand;
          }
          return qVec::quantize(dx_vec_vec, o_scale, o_zp, o_inv_scale);
        });
  });
}

void qprelu_out_kernel(Tensor& out,
                              const Tensor& qx,
                              const Tensor& qw) {
  int32_t i_zp = static_cast<int32_t>(qx.q_zero_point());
  float i_scale = static_cast<float>(qx.q_scale());

  int32_t w_zp = static_cast<int32_t>(qw.q_zero_point());
  float w_scale = static_cast<float>(qw.q_scale());

  int32_t o_zp = static_cast<int32_t>(out.q_zero_point());
  float o_scale = static_cast<float>(out.q_scale());
  float o_inv_scale = 1.0f / o_scale;

  float multiplier = i_scale * w_scale * o_inv_scale;

  int64_t input_ndim = qx.dim();
  TORCH_CHECK(input_ndim > 0, "qprelu: zero-dim input tensor is not allowed.");

  // This logic is present in at::prelu and repeated here, as this path can be
  // hit via quantized::prelu, which is registered under quantized/cpu/qprelu.cpu
  auto qw_nd = qw;
  if (input_ndim != qw_nd.dim()) {
    DimVector dim_w(input_ndim, 1);
    if (input_ndim > 1) {
      dim_w[1] = qw.numel();
    }
    // This will always be a view in CPU/CUDA, but some backends
    // like MKLDNN do not support views
    qw_nd = qw_nd.reshape(dim_w);
  }

  auto iter = TensorIteratorConfig()
    .add_output(out)
    .add_input(qx)
    .add_input(qw_nd)
    .build();

  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qprelu", [&] {
    using qVec = Vectorized<scalar_t>;
    qVec i_zp_vec = qVec(static_cast<scalar_t>(i_zp));
    qVec w_zp_vec = qVec(static_cast<scalar_t>(w_zp));

    // Quantized one as weight
    auto qw_one = at::native::quantize_val<scalar_t>(w_scale, w_zp, 1.0f);
    qVec vec_qw_one = qVec(qw_one);
    auto vec_qw_one_sub_zp = vec_qw_one.widening_subtract(w_zp_vec)[0];
    int32_t qw_one_sub_zp = qw_one.val_ - w_zp;

    cpu_kernel_vec(
      iter,
      [=](scalar_t val_qx, scalar_t val_qw) -> scalar_t {
        int32_t qx_pos = std::max(static_cast<int32_t>(val_qx.val_), i_zp);
        int32_t qx_neg = std::min(static_cast<int32_t>(val_qx.val_), i_zp);
        int32_t qx_pos_sub_zp = qx_pos - i_zp;
        int32_t qx_neg_sub_zp = qx_neg - i_zp;
        int32_t qw_sub_zp = val_qw.val_ - w_zp;
        auto qy_sub_zp = qx_pos_sub_zp * qw_one_sub_zp + qx_neg_sub_zp * qw_sub_zp;
        return at::native::requantize_from_int<scalar_t>(
            multiplier, o_zp, qy_sub_zp);
      },
      [=](qVec vec_qx, qVec vec_qw) -> qVec {
        auto vec_qx_pos = vec_qx.maximum(i_zp_vec);
        auto vec_qx_neg = vec_qx.minimum(i_zp_vec);
        qVec::int_vec_return_type qx_pos_sub_zp = vec_qx_pos.widening_subtract(i_zp_vec);
        qVec::int_vec_return_type qx_neg_sub_zp = vec_qx_neg.widening_subtract(i_zp_vec);
        qVec::int_vec_return_type qw_sub_zp = vec_qw.widening_subtract(w_zp_vec);
        qVec::int_vec_return_type qy_sub_zp;
        for (const auto i : c10::irange(qVec::int_num_vecs())) {
          qy_sub_zp[i] = qx_pos_sub_zp[i] * vec_qw_one_sub_zp + qx_neg_sub_zp[i] * qw_sub_zp[i];
        }
        return qVec::requantize_from_int(qy_sub_zp, multiplier, o_zp);
      });
  });

}

void qgelu_kernel(const Tensor& qx, Tensor& qy, GeluType approximate) {
  int64_t zero_point = qx.q_zero_point();
  float scale = static_cast<float>(qx.q_scale());
  auto scale_vec = Vectorized<float>(scale);
  auto zero_point_vec = Vectorized<float>(zero_point);
  auto scale_neg_zp_premul_vec = scale_vec * zero_point_vec.neg();
  int64_t output_zero_point = zero_point;
  float output_scale = scale;
  float inv_output_scale = 1.0 / output_scale;
  const auto kAlphaVec = Vectorized<float>(M_SQRT1_2);
  const auto kBetaVec = Vectorized<float>(M_SQRT2 * M_2_SQRTPI * 0.5);
  const auto kKappaVec = Vectorized<float>(0.044715);
  const auto kOneVec = Vectorized<float>(1);
  const auto kPointFiveVec = Vectorized<float>(0.5);

  if (approximate == GeluType::Tanh) {
    AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qgelu", [&]() {
      qy = at::_empty_affine_quantized(
          qx.sizes(),
          at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
          output_scale,
          output_zero_point,
          std::nullopt);
      auto iter = TensorIterator::unary_op(qy, qx);

      using Vec = Vectorized<scalar_t>;
      cpu_kernel_vec(
          iter,
          [&](scalar_t value_qx) -> scalar_t {
            const auto value_dx =
                at::native::dequantize_val(scale, zero_point, value_qx);

            const auto kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
            const auto kKappa = 0.044715;
            const auto x_cube = value_dx * value_dx * value_dx;
            const auto inner = kBeta * (value_dx + kKappa * x_cube);
            const auto value_dy = 0.5 * value_dx * (1.0 + std::tanh(inner));

            return at::native::quantize_val<scalar_t>(
                output_scale, output_zero_point, value_dy);
          },
          [&](Vec value_qx) -> Vec {
            auto value_dx = value_qx.dequantize(
                scale_vec, zero_point_vec, scale_neg_zp_premul_vec);
            for (auto & value : value_dx) {
              auto value_cube = value * value * value;
              auto inner = kBetaVec * (value + kKappaVec * value_cube);
              value = kPointFiveVec * value * (kOneVec + inner.tanh());
            }
            return Vec::quantize(
                value_dx, output_scale, output_zero_point, inv_output_scale);
          });
    });
  } else {
    AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qgelu", [&]() {
      qy = at::_empty_affine_quantized(
          qx.sizes(),
          at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
          output_scale,
          output_zero_point,
          std::nullopt);
      auto iter = TensorIterator::unary_op(qy, qx);

      using Vec = Vectorized<scalar_t>;
      cpu_kernel_vec(
          iter,
          [&](scalar_t value_qx) -> scalar_t {
            const auto value_dx =
                at::native::dequantize_val(scale, zero_point, value_qx);
            const auto value_dy =
                value_dx * 0.5 * (1 + std::erf(value_dx * M_SQRT1_2));
            return at::native::quantize_val<scalar_t>(
                output_scale, output_zero_point, value_dy);
          },
          [&](Vec value_qx) -> Vec {
            auto value_dx = value_qx.dequantize(
                scale_vec, zero_point_vec, scale_neg_zp_premul_vec);
            for (auto & value : value_dx) {
              value = value * kPointFiveVec * (kOneVec + (value * kAlphaVec).erf());
            }
            return Vec::quantize(
                value_dx, output_scale, output_zero_point, inv_output_scale);
          });
    });
  }
}


void qsigmoid_kernel(
    const Tensor& qx, Tensor& qy, double output_scale, int64_t output_zero_point ) {
  int64_t zero_point = qx.q_zero_point();
  float scale = static_cast<float>(qx.q_scale());
  auto scale_vec = Vectorized<float>(scale);
  auto zero_point_vec = Vectorized<float>(zero_point);

  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qsigmoid", [&]() {
    float inv_output_scale = 1.0 / output_scale;

    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
        output_scale,
        output_zero_point,
        std::nullopt);
    auto iter = TensorIterator::unary_op(qy, qx);

    using Vec = Vectorized<scalar_t>;
    cpu_kernel_vec(
        iter,
        [&](scalar_t value_qx) -> scalar_t {
          const auto value_dx =
              at::native::dequantize_val(scale, zero_point, value_qx);
          const auto value_dy = 1.0f / (1.0 + std::exp((-value_dx)));
          return at::native::quantize_val<scalar_t>(
              output_scale, output_zero_point, value_dy);
        },
        [&](Vec value_qx) -> Vec {
          auto value_dx = value_qx.dequantize(scale_vec, zero_point_vec);
          for (auto & value : value_dx) {
            value = value.neg();
            value = value.exp();
            value = Vectorized<float>(1.0f) + value;
            value = value.reciprocal();
          }
          return Vec::quantize(
              value_dx, output_scale, output_zero_point, inv_output_scale);
        });
  });
}

void qhardsigmoid_kernel(const Tensor& qx, Tensor& qy) {
  int64_t zero_point = qx.q_zero_point();
  float scale = static_cast<float>(qx.q_scale());
  auto scale_vec = Vectorized<float>(scale);
  auto zero_point_vec = Vectorized<float>(zero_point);
  auto scale_neg_zp_premul_vec = scale_vec * zero_point_vec.neg();

  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qhardsigmoid", [&]() {

    // - Output scale is set to 1.0 / 2^(BIT_NUM)
    float output_scale = 0.00390625;  // 1.0 / 2^8
    if (SCALAR_TYPE == at::kQInt32) {
      output_scale = 2.3283064365386963e-10;  // 1.0 / 2^32
    }
    float inv_output_scale = 1.0 / output_scale;

    // The default zero-point is zero.  As a one-off optimization for
    // kQInt8, we set the zero-point to -128 to maximize precision in the
    // [0, 1] output range. kQInt32 can be handled in a future PR if needed.
    int64_t output_zero_point = 0;
    if (SCALAR_TYPE == at::kQInt8) {
      output_zero_point = -128;
    }

    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE),
        output_scale,
        output_zero_point,
        qx.suggest_memory_format());
    auto iter = TensorIterator::unary_op(qy, qx);

    using qVec = Vectorized<scalar_t>;
    using fVec = Vectorized<float>;
    fVec kZeroVec(0.0f);
    fVec kThreeVec(3.0f);
    fVec kSixVec(6.0f);

    // Naive implementation: uses dequantize/execute/quantize routine
    cpu_kernel_vec(
        iter,
        [&](scalar_t qx) -> scalar_t {
          auto x = at::native::dequantize_val(scale, zero_point, qx);
          const auto y = std::min(std::max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
          return at::native::quantize_val<scalar_t>(
              output_scale, output_zero_point, y);
        },
        [&](qVec value_qx) -> qVec {
          auto value_dx = value_qx.dequantize(
              scale_vec, zero_point_vec, scale_neg_zp_premul_vec);
          for (auto & value : value_dx) {
            value =
                vec::minimum(
                    vec::maximum(value + kThreeVec, kZeroVec),
                    kSixVec) /
                kSixVec;
          }
          return qVec::quantize(
              value_dx, output_scale, output_zero_point, inv_output_scale);
        });
  });
}

void qclamp_kernel(
    const Tensor& qx,
    const Scalar& min_scalar,
    const Scalar& max_scalar,
    Tensor& qy) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qclamp", [&]() {
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point(),
        std::nullopt);
    using Vec = Vectorized<scalar_t>;
    auto iter = TensorIterator::unary_op(qy, qx);
    auto min = min_scalar.to<float>();
    auto max = max_scalar.to<float>();
    scalar_t min_q = at::native::quantize_val<scalar_t>(
        qx.q_scale(), qx.q_zero_point(), min);
    scalar_t max_q = at::native::quantize_val<scalar_t>(
        qx.q_scale(), qx.q_zero_point(), max);
    auto min_vec = Vec(min_q);
    auto max_vec = Vec(max_q);
    cpu_kernel_vec(
        iter,
        [&](scalar_t value) -> scalar_t {
          underlying_t min_clamped =
              std::max<underlying_t>(value.val_, min_q.val_);
          return scalar_t(std::min<underlying_t>(min_clamped, max_q.val_));
        },
        [&](Vec val) -> Vec {
          auto min_clamped = val.maximum(min_vec);
          return min_clamped.minimum(max_vec);
        });
  });
}

void qclamp_min_kernel(const Tensor& qx, const Scalar& min_scalar, Tensor& qy) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qclamp", [&]() {
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU)
            .dtype(SCALAR_TYPE)
            .memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point(),
        std::nullopt);
    using Vec = Vectorized<scalar_t>;
    auto iter = TensorIterator::unary_op(qy, qx);
    auto min = min_scalar.to<float>();
    scalar_t min_q = at::native::quantize_val<scalar_t>(
        qx.q_scale(), qx.q_zero_point(), min);
    auto min_vec = Vec(min_q);
    cpu_kernel_vec(
        iter,
        [&](scalar_t value) -> scalar_t {
          return scalar_t(std::max<underlying_t>(value.val_, min_q.val_));
        },
        [&](Vec val) -> Vec { return val.maximum(min_vec); });
  });
}

void qclamp_max_kernel(const Tensor& qx, const Scalar& max_scalar, Tensor& qy) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qclamp", [&]() {
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU)
            .dtype(SCALAR_TYPE)
            .memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point(),
        std::nullopt);
    using Vec = Vectorized<scalar_t>;
    auto iter = TensorIterator::unary_op(qy, qx);
    auto max = max_scalar.to<float>();
    scalar_t max_q = at::native::quantize_val<scalar_t>(
        qx.q_scale(), qx.q_zero_point(), max);
    auto max_vec = Vec(max_q);
    cpu_kernel_vec(
        iter,
        [&](scalar_t value) -> scalar_t {
          return scalar_t(std::min<underlying_t>(value.val_, max_q.val_));
        },
        [&](Vec val) -> Vec { return val.minimum(max_vec); });
  });
}

void qthreshold_kernel(
  // TODO: For future tasks, since output quantization parameters are set equal to
  // the input ones, it might make sense to implement this completely in the
  // quantized domain.
   const Tensor& qx,
   const Scalar& threshold_scalar,
   const Scalar& value_scalar,
   Tensor& qy) {

  // defines input and output scales and zero_points
  int64_t input_zero_point = qx.q_zero_point();
  float input_scale = static_cast<float>(qx.q_scale());
  int64_t output_zero_point = qy.q_zero_point();
  float output_scale = static_cast<float>(qy.q_scale());
  float inv_output_scale = static_cast<float>(1.0 / output_scale);

  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qthreshold", [&]() {
    qy = at::_empty_affine_quantized(
      qx.sizes(),
      at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
      qx.q_scale(),
      qx.q_zero_point(),
      std::nullopt);

    // vectorized
    using Vec = Vectorized<float>;
    using qVec = Vectorized<scalar_t>;
    // defines the iterator
    auto iter = TensorIterator::unary_op(qy, qx);
    // defines the vectorized versions
    Vec input_scale_vec = Vec(input_scale);
    Vec input_zero_point_vec = Vec(input_zero_point);
    Vec input_scale_neg_zp_premul_vec = input_scale_vec * input_zero_point_vec.neg();
    // defines the floating-point versions of threshold and value
    float threshold_float = threshold_scalar.to<float>();
    float value_float = value_scalar.to<float>();
    Vec threshold_vec = Vec(threshold_float);
    Vec value_vec = Vec(value_float);

    // Naive implementation: uses dequantize/execute/quantize routine
    cpu_kernel_vec(
        iter,
        [&](scalar_t value_qx) -> scalar_t {
          // dequantize
          const auto x = at::native::dequantize_val(input_scale, input_zero_point, value_qx);
          // Applies the Threshold operation
          const auto y = x > threshold_float ? x : value_float;
          // quantize
          return at::native::quantize_val<scalar_t>(output_scale, output_zero_point, y);
        },
        [&](qVec value_qx) -> qVec {
          // dequantize
          auto dx_vec = value_qx.dequantize(
            input_scale_vec, input_zero_point_vec, input_scale_neg_zp_premul_vec);
          for (auto & value : dx_vec) {
            // check if any elements are below threshold
            const auto cmp_to_threshold = value > threshold_vec;
            if (cmp_to_threshold.zero_mask()) {
              // blend
              value = Vec::blendv(value_vec, value, cmp_to_threshold);
            }
          }
          // quantize
          return qVec::quantize(dx_vec, output_scale, output_zero_point, inv_output_scale);
        });
  });
}


void qhardswish_kernel(const Tensor& qx, Tensor& qy) {
  const auto i_scale = qx.q_scale();
  const auto i_zero_point = qx.q_zero_point();

  const auto o_scale = qy.q_scale();
  const auto o_zero_point = qy.q_zero_point();
  const float o_inv_scale = static_cast<float>(1.0 / o_scale);

  using fVec = Vectorized<float>;
  fVec i_scale_vec(i_scale);
  fVec i_zero_point_vec(i_zero_point);
  fVec i_scale_neg_zp_premul_vec = i_scale_vec * i_zero_point_vec.neg();
  fVec zero_vec(0.0f);
  fVec three_vec(3.0f);
  fVec six_vec(6.0f);

  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qhardswish", [&]() {
    using qVec = Vectorized<scalar_t>;
    auto iter = TensorIterator::unary_op(qy, qx);
    cpu_kernel_vec(
        iter,
        [&](scalar_t value) -> scalar_t {
          const auto x =
              at::native::dequantize_val(i_scale, i_zero_point, value);
          const auto y = x * std::min(std::max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
          return at::native::quantize_val<scalar_t>(o_scale, o_zero_point, y);
        },
        [&](qVec value) -> qVec {
          auto value_dx = value.dequantize(i_scale_vec, i_zero_point_vec,
                                           i_scale_neg_zp_premul_vec);
          for (auto & value : value_dx) {
            value = value * vec::minimum(
              vec::maximum(value + three_vec, zero_vec),
              six_vec
            ) / six_vec;
          }
          return qVec::quantize(value_dx, o_scale, o_zero_point, o_inv_scale);
        });
  });
}


void qtanh_kernel(const Tensor& qx, Tensor& qy) {
  int64_t zero_point = qx.q_zero_point();
  float scale = static_cast<float>(qx.q_scale());
  auto scale_vec = Vectorized<float>(scale);
  auto zero_point_vec = Vectorized<float>(zero_point);
  auto scale_neg_zp_premul_vec = scale_vec * zero_point_vec.neg();

  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qtanh", [&]() {
    // Naive implementation: uses dequantize/execute/quantize routine
    // - Output scale is set to 2.0 / 2^(BIT_NUM)
    // - For signed types output zero point is set to 0
    // - For unsigned types output zero point is set to (qmax + qmin) / 2.0
    float output_scale = 0.0078125;  // 2.0 / 512
    int64_t output_zero_point = 0;
    if (SCALAR_TYPE == at::kQInt32) {
      output_scale = 4.656612873077393e-10;  // 2.0 / 2^32
    } else if (SCALAR_TYPE == at::kQUInt8) {
      output_zero_point = 128;
    }
    float inv_output_scale = 1.0 / output_scale;

    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
        output_scale,
        output_zero_point,
        std::nullopt);
    auto iter = TensorIterator::unary_op(qy, qx);

    using Vec = Vectorized<scalar_t>;
    cpu_kernel_vec(
        iter,
        [&](scalar_t value_qx) -> scalar_t {
          const auto value_dx =
              at::native::dequantize_val(scale, zero_point, value_qx);
          return at::native::quantize_val<scalar_t>(
              output_scale, output_zero_point, std::tanh(value_dx));
        },
        [&](Vec value_qx) -> Vec {
          const auto value_dx = value_qx.dequantize(
              scale_vec, zero_point_vec, scale_neg_zp_premul_vec);
          Vec::float_vec_return_type retvals;
          for (const auto idx : c10::irange(Vec::float_num_vecs())) {
            retvals[idx] = value_dx[idx].tanh();
          }
          return Vec::quantize(
              retvals, output_scale, output_zero_point, inv_output_scale);
        });
  });
}

void qelu_kernel(
    const Tensor& qx,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    Tensor& qy) {
  // scale and input_scale arguments refer to a generalized ELU formula
  // if x >= 0, ELU(x) = x * scale
  // if x <= 0, ELU(x) = (exp(x * input_scale) - 1) * scale
  // in the normal ELU formula, both are equal to 1
  // they are NOT related to the quantization scale term

  int64_t i_zp = qx.q_zero_point();
  float i_scale = static_cast<float>(qx.q_scale());

  // In a future PR, we can improve on output scale and zero_point
  // selection.
  int64_t o_zp = qy.q_zero_point();
  float o_scale = static_cast<float>(qy.q_scale());
  float inv_o_scale = static_cast<float>(1.0 / o_scale);

  float alpha_float = alpha.to<float>();
  float scale_coef = scale.to<float>();
  float input_scale_coef = input_scale.to<float>();

  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qelu_kernel", [&] {

    auto iter = TensorIterator::unary_op(qy, qx);

    // vectorized
    using Vec = Vectorized<float>;
    using qVec = Vectorized<scalar_t>;

    Vec zero_vec = Vec(0.0f);
    Vec one_vec = Vec(1.0f);
    Vec alpha_vec = Vec(alpha_float);
    Vec scale_coef_vec = Vec(scale_coef);
    Vec input_scale_coef_vec = Vec(input_scale_coef);
    Vec i_scale_vec = Vec(i_scale);
    Vec i_zero_point_vec = Vec(i_zp);
    Vec i_scale_neg_zp_premul_vec = i_scale_vec * i_zero_point_vec.neg();

    cpu_kernel_vec(
      iter,
      [&](scalar_t value_qx) -> scalar_t {
        // dequantize
        const auto x = at::native::dequantize_val(i_scale, i_zp, value_qx);
        // ELU
        const auto y = x >= 0
          ? x * scale_coef
          : ((std::exp(x * input_scale_coef) - 1) * alpha_float * scale_coef);

        // quantize
        return at::native::quantize_val<scalar_t>(o_scale, o_zp, y);
      },
      [&](qVec value_qx) -> qVec {
        // dequantize
        auto dx_vec_vec = value_qx.dequantize(i_scale_vec, i_zero_point_vec,
                                            i_scale_neg_zp_premul_vec);
        for (auto & value : dx_vec_vec) {
          // quickly check if any elements are below zero
          const auto cmp_to_zero = value > zero_vec;

          if (cmp_to_zero.zero_mask()) {

            Vec dx_vec_copy_neg_elu = value * one_vec;
            // calculate the negative part of ELU on the copy
            dx_vec_copy_neg_elu = dx_vec_copy_neg_elu * input_scale_coef_vec;
            dx_vec_copy_neg_elu = dx_vec_copy_neg_elu.exp();
            dx_vec_copy_neg_elu = dx_vec_copy_neg_elu - one_vec;
            dx_vec_copy_neg_elu = dx_vec_copy_neg_elu * alpha_vec;
            // blend
            value = Vec::blendv(dx_vec_copy_neg_elu, value,
                                        value > zero_vec);
          }

          value = value * scale_coef_vec;
        }
        // quantize
        return qVec::quantize(dx_vec_vec, o_scale, o_zp, inv_o_scale);
      }
    );

  });
}

// Note: out is assumed to be the same size as self and other.
// Note: Addition is only supported when self and out are of the same dtype.
// Note: other is already assumed to be in int32, i.e., it's
// round(float/self_scale)
template <bool ReLUFused = false>
void qadd_scalar_kernel(Tensor& out, const Tensor& self, const Scalar& other) {
  int64_t zero_point = out.q_zero_point();
  float scale = static_cast<float>(out.q_scale());
  float inv_scale = 1.0f / scale;
  int64_t self_zero_point = self.q_zero_point();
  float self_scale = static_cast<float>(self.q_scale());

  float multiplier = self_scale * inv_scale;

  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "qadd_scalar", [&]() {
    using Vec = Vectorized<scalar_t>;
    auto iter = TensorIterator::unary_op(out, self);
    auto other_val = other.to<int32_t>();
    auto other_vec = Vectorized<c10::qint32>(static_cast<c10::qint32>(other_val));
    cpu_kernel_vec(
        iter,
        [&](scalar_t a) -> scalar_t {
          int32_t a_sub_z = static_cast<int32_t>(a.val_) -
              static_cast<int32_t>(self_zero_point);
          int32_t c = a_sub_z + other_val;
          scalar_t res = at::native::requantize_from_int<scalar_t>(
              multiplier, zero_point, c);
          if constexpr (ReLUFused) {
            res.val_ = std::max<scalar_t::underlying>(res.val_, zero_point);
          }
          return res;
        },
        [&](Vec a) -> Vec {
          Vec::int_vec_return_type a_sub_z =
              a.widening_subtract(Vec(static_cast<scalar_t>(self_zero_point)));
          Vec::int_vec_return_type c;
          for (const auto i : c10::irange(Vec::int_num_vecs())) {
            c[i] = a_sub_z[i] + other_vec;
          }
          Vec rv = Vec::requantize_from_int(c, multiplier, zero_point);
          if constexpr (ReLUFused) {
            rv = rv.maximum(Vec(static_cast<scalar_t>(zero_point)));
          }
          return rv;
        });
  });
}
// Note: out is assumed to be the same size as self and other.
// Note: Addition is only supported when self, other, out are of the same dtype.
template <bool ReLUFused = false>
void qadd_kernel(Tensor& out, const Tensor& self, const Tensor& other) {
  int64_t zero_point = out.q_zero_point();
  float scale = static_cast<float>(out.q_scale());
  float inv_scale = 1.0f / scale;
  int64_t self_zero_point = self.q_zero_point();
  float self_scale = static_cast<float>(self.q_scale());
  int64_t other_zero_point = other.q_zero_point();
  float other_scale = static_cast<float>(other.q_scale());

  // Broadcast out the parameters here to amortize out that cost across
  // loop iterations.
  // TODO: we can optimize dequantization by doing a premultiplication
  // of the zero point by scale and doing FMA on scale*x_q - (scale*zero_point)
  auto self_zero_point_vec = Vectorized<float>(self_zero_point);
  auto self_scale_vec = Vectorized<float>(self_scale);
  auto other_zero_point_vec = Vectorized<float>(other_zero_point);
  auto other_scale_vec = Vectorized<float>(other_scale);

  auto self_scale_neg_zp_premul_vec = self_scale_vec * self_zero_point_vec.neg();
  auto other_scale_zp_premul_vec = other_scale_vec * other_zero_point_vec.neg();

  auto iter = TensorIterator::borrowing_binary_op(out, self, other);

  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qadd", [&]() {
    using Vec = Vectorized<scalar_t>;
    cpu_kernel_vec(
        iter,
        [&](scalar_t a, scalar_t b) -> scalar_t {
          const auto da =
              at::native::dequantize_val(self_scale, self_zero_point, a);
          const auto db =
              at::native::dequantize_val(other_scale, other_zero_point, b);
          float c = da + db;
          if (ReLUFused) {
            c = std::max<float>(c, 0.0);
          }
          return at::native::quantize_val<scalar_t>(scale, zero_point, c);
        },
        [&](Vec a, Vec b) -> Vec {
          const auto da = a.dequantize(
              self_scale_vec, self_zero_point_vec, self_scale_neg_zp_premul_vec);
          const auto db = b.dequantize(
              other_scale_vec, other_zero_point_vec, other_scale_zp_premul_vec);
          Vec::float_vec_return_type retvals;
          for (const auto i : c10::irange(Vec::float_num_vecs())) {
            auto c = da[i] + db[i];
            if constexpr (ReLUFused) {
              c = vec::maximum(c, Vectorized<float>(0.0f));
            }
            retvals[i] = c;
          }
          // TODO: fbgemm::Quantize doesn't support taking in the
          // pre-broadcasted parameters. We might be able to save some cycles by
          // enabling that in the API.
          // TODO: specialize fbgemm::Quantize for a single vector and make it
          // inlineable. This could help with interleaving as suggested by the
          // TensorIterator implementations
          auto rv = Vec::quantize(retvals, scale, zero_point, inv_scale);
          return rv;
        });
  });
}

// Note: out is assumed to be the same size as self and other.
// Note: Multiplication is only supported when self, other, out are of the same
// dtype.
template <bool ReLUFused = false>
void qmul_kernel(Tensor& out, const Tensor& self, const Tensor& other) {
  int64_t zero_point = out.q_zero_point();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  float scale = out.q_scale();
  float inv_scale = 1.0f / scale;
  int64_t self_zero_point = self.q_zero_point();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  float self_scale = self.q_scale();
  int64_t other_zero_point = other.q_zero_point();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  float other_scale = other.q_scale();

  float multiplier = self_scale * other_scale * inv_scale;

  auto iter = TensorIterator::borrowing_binary_op(out, self, other);

  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qmul", [&]() {
    using Vec = Vectorized<scalar_t>;
    cpu_kernel_vec(
        iter,
        [&](scalar_t a, scalar_t b) -> scalar_t {
          int32_t a_sub_z = static_cast<int32_t>(a.val_) -
              static_cast<int32_t>(self_zero_point);
          int32_t b_sub_z = static_cast<int32_t>(b.val_) -
              static_cast<int32_t>(other_zero_point);
          int32_t c = a_sub_z * b_sub_z;
          scalar_t res = at::native::requantize_from_int<scalar_t>(
              multiplier, zero_point, c);
          if constexpr (ReLUFused) {
            res.val_ = std::max<scalar_t::underlying>(res.val_, zero_point);
          }
          return res;
        },
        [&](Vec a, Vec b) -> Vec {
          Vec::int_vec_return_type a_sub_zp =
              a.widening_subtract(Vec(static_cast<scalar_t>(self_zero_point)));
          Vec::int_vec_return_type b_sub_zp =
              b.widening_subtract(Vec(static_cast<scalar_t>(other_zero_point)));
          Vec::int_vec_return_type c;
          for (const auto i : c10::irange(Vec::int_num_vecs())) {
            c[i] = a_sub_zp[i] * b_sub_zp[i];
          }
          Vec rv = Vec::requantize_from_int(c, multiplier, zero_point);
          if constexpr (ReLUFused) {
            rv = rv.maximum(Vec(static_cast<scalar_t>(zero_point)));
          }
          return rv;
        });
  });
}

template <typename scalar_t, typename scalar_t_underlying>
void _qmaxpool_2d_nhwc_kernel(
    const Tensor& qx,
    int64_t iC, // input/output channels
    int64_t iH,
    int64_t iW, // input sizes
    int64_t oH,
    int64_t oW, // output sizes
    int64_t kH,
    int64_t kW, // kernel size
    int64_t sH,
    int64_t sW, // strides
    int64_t pH,
    int64_t pW, // padding
    int64_t dH,
    int64_t dW, // dilation
    Tensor& qy) {
    scalar_t* idata = static_cast<scalar_t*>(qx.data_ptr());
    scalar_t* odata = static_cast<scalar_t*>(qy.data_ptr());

    int64_t nBatch = qx.size(0);
    at::parallel_for(0, nBatch * oH * oW, 0, [&](int64_t begin, int64_t end) {
      int64_t b{0}, row{0}, col{0};
      data_index_init(begin, b, nBatch, row, oH, col, oW);

      for (const auto i : c10::irange(begin, end)) {
        auto* i_p = reinterpret_cast<scalar_t_underlying*>(idata + b * iW * iH * iC);
        auto* o_p = reinterpret_cast<scalar_t_underlying*>(odata + i * iC);

        // Loop over reduction block
        int64_t h_start = row * sH - pH;
        int64_t w_start = col * sW - pW;
        int64_t h_end = std::min(h_start + (kH - 1) * dH + 1, iH);
        int64_t w_end = std::min(w_start + (kW - 1) * dW + 1, iW);
        while (h_start < 0)
          h_start += dH;
        while (w_start < 0)
          w_start += dW;

        int64_t c = 0;

        // Interleaved vector loop 4x
        constexpr auto vec_width = Vectorized<scalar_t>::size();
        for (; c + 4 * vec_width <= iC; c += 4 * vec_width) {
          Vectorized<scalar_t> acc{
              scalar_t(std::numeric_limits<scalar_t_underlying>::lowest())};
          // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
          Vectorized<scalar_t> accs[4] = {acc, acc, acc, acc};
          int64_t tcntr = 0;
          int64_t x, y;
          for (y = h_start; y < h_end; y += dH) {
            for (x = w_start; x < w_end; x += dW) {
              for (const auto i : c10::irange(4)) {
                tcntr = y * iW + x;
                auto vals = Vectorized<scalar_t>::loadu(
                    i_p + tcntr * iC + c + Vectorized<scalar_t>::size() * i);
                accs[i] = vec::maximum(accs[i], vals);
              }
            } // for x
          } // for y
          for (const auto i : c10::irange(4)) {
            accs[i].store(o_p + c + Vectorized<scalar_t>::size() * i);
          }
        } // for c

        // Vector loop
        for (; c + vec_width <= iC; c += vec_width) {
          Vectorized<scalar_t> acc{
              scalar_t(std::numeric_limits<scalar_t_underlying>::lowest())};
          int64_t tcntr = 0;
          int64_t x, y;
          for (y = h_start; y < h_end; y += dH) {
            for (x = w_start; x < w_end; x += dW) {
              tcntr = y * iW + x;
              auto vals = Vectorized<scalar_t>::loadu(i_p + tcntr * iC + c);
              acc = vec::maximum(acc, vals);
            } // for x
          } // for y
          acc.store(o_p + c);
        } // for c

        for (; c < iC; ++c) {
          auto max_val = std::numeric_limits<scalar_t_underlying>::lowest();
          int64_t tcntr = 0;
          int64_t x, y;
          for (y = h_start; y < h_end; y += dH) {
            for (x = w_start; x < w_end; x += dW) {
              tcntr = y * iW + x;
              auto val = *(i_p + tcntr * iC + c);
              max_val = std::max(max_val, val);
            } // for x
          } // for y

          o_p[c] = max_val;
        } // for c

        data_index_step(b, nBatch, row, oH, col, oW);
      }
    });
}

void qmaxpool_2d_nhwc_kernel(
    const Tensor& qx,
    int64_t iC, // input/output channels
    int64_t iH,
    int64_t iW, // input sizes
    int64_t oH,
    int64_t oW, // output sizes
    int64_t kH,
    int64_t kW, // kernel size
    int64_t sH,
    int64_t sW, // strides
    int64_t pH,
    int64_t pW, // padding
    int64_t dH,
    int64_t dW, // dilation
    Tensor& qy) {
  if (qx.scalar_type() == ScalarType::Byte) {
    AT_DISPATCH_INTEGRAL_TYPES(qx.scalar_type(), "max_pool2d_nhwc", [&]() {
      _qmaxpool_2d_nhwc_kernel<scalar_t, scalar_t>(qx, iC, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, qy);
    });
  } else {
    AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "max_pool2d_nhwc", [&]() {
      _qmaxpool_2d_nhwc_kernel<scalar_t, scalar_t::underlying>(qx, iC, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, qy);
    });
  }
}

void qmaxpool_3d_nthwc_kernel(
    const Tensor& qx,
    int64_t iC, // input/output channels
    int64_t iT,
    int64_t iH,
    int64_t iW, // input sizes
    int64_t oT,
    int64_t oH,
    int64_t oW, // output sizes
    int64_t kT,
    int64_t kH,
    int64_t kW, // kernel size
    int64_t sT,
    int64_t sH,
    int64_t sW, // strides
    int64_t pT,
    int64_t pH,
    int64_t pW, // padding
    int64_t dT,
    int64_t dH,
    int64_t dW, // dilation
    Tensor& qy) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "max_pool3d_nthwc", [&]() {
    scalar_t* idata = static_cast<scalar_t*>(qx.data_ptr());
    scalar_t* odata = static_cast<scalar_t*>(qy.data_ptr());
    int64_t nBatch = qx.size(0);
    at::parallel_for(0, nBatch * oT * oH * oW, 0, [&](int64_t begin, int64_t end) {
      int64_t b{0}, time{0}, row{0}, col{0};

      data_index_init(begin, b, nBatch, time, oT, row, oH, col, oW);

      for (const auto i : c10::irange(begin, end)) {
        auto* i_p = reinterpret_cast<scalar_t::underlying*>(idata + b * iT * iW * iH * iC);
        auto* o_p = reinterpret_cast<scalar_t::underlying*>(odata + i * iC);

        // Loop over reduction block
        int64_t t_start = time * sT - pT;
        int64_t h_start = row * sH - pH;
        int64_t w_start = col * sW - pW;
        int64_t t_end = std::min(t_start + (kT - 1) * dT + 1, iT);
        int64_t h_end = std::min(h_start + (kH - 1) * dH + 1, iH);
        int64_t w_end = std::min(w_start + (kW - 1) * dW + 1, iW);
        while (t_start < 0)
          t_start += dT;
        while (h_start < 0)
          h_start += dH;
        while (w_start < 0)
          w_start += dW;

        int64_t c = 0;
        constexpr auto vec_width = Vectorized<scalar_t>::size();
        // Vector loop
        for (; c + vec_width <= iC; c += vec_width) {
          Vectorized<scalar_t> acc{
              scalar_t(std::numeric_limits<scalar_t::underlying>::lowest())};
          int64_t tcntr = 0;
          int64_t t, x, y;
          for (t = t_start; t < t_end; t += dT) {
            for (y = h_start; y < h_end; y += dH) {
              for (x = w_start; x < w_end; x += dW) {
                tcntr = t * iH * iW + y * iW + x;
                auto vals = Vectorized<scalar_t>::loadu(i_p + tcntr * iC + c);
                acc = vec::maximum(acc, vals);
              } // for x
            } // for y
          } // for t
          acc.store(o_p + c);
        } // for c

        for (; c < iC; ++c) {
          auto max_val = std::numeric_limits<scalar_t::underlying>::lowest();
          int64_t tcntr = 0;
          int64_t t, x, y;
          for (t = t_start; t < t_end; t += dT) {
            for (y = h_start; y < h_end; y += dH) {
              for (x = w_start; x < w_end; x += dW) {
                tcntr = t * iH * iW + y * iW + x;
                auto val = *(i_p + tcntr * iC + c);
                max_val = std::max(max_val, val);
              } // for x
            } // for y
          } // for t
          o_p[c] = max_val;
        } // for c
        data_index_step(b, nBatch, time, oT, row, oH, col, oW);
      }

    });

  });
}

template <typename T>
void do_avg_pool_nhwc_on_AVX_n(
    const typename T::underlying* i_p,
    typename T::underlying* o_p,
    int& c_start,
    int input_zero_point_m_size,
    int output_zero_point,
    float multiplier,
    int dstart,
    int dend,
    int hstart,
    int hend,
    int wstart,
    int wend,
    int dsize,
    int hsize,
    int wsize,
    int csize) {
#if (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)) && !defined(_MSC_VER)
  // buffer for channel accumulator, used to interchange channel-loop
  // to inner-most, so that memory access of the input tensor data is
  // continuous.
#ifdef CPU_CAPABILITY_AVX2
  constexpr int cb_size = 16;
#else
  constexpr int cb_size = 8;
#endif
  constexpr int vec_width = Vectorized<T>::size() / 4;
  constexpr int cb_step = cb_size * vec_width;
  Vectorized<int32_t> acc_buffer[cb_size];
  Vectorized<float> acc_buffer_fp[cb_size];

#ifdef CPU_CAPABILITY_AVX2
  if (vec_width == 8) {
#else
  if (vec_width == 16) {
#endif
    for (int c = c_start; c < csize; c += cb_step) {
      int cend = std::min(cb_size, (csize - c) / vec_width);
      // initialize loop
      for (const auto ic : c10::irange(cend)) {
        acc_buffer[ic] = Vectorized<int32_t>(input_zero_point_m_size);
      }
      // compute loop
      for (const auto id : c10::irange(dstart, dend)) {
        for (const auto ih : c10::irange(hstart, hend)) {
          for (const auto iw : c10::irange(wstart, wend)) {
            const int i_idx =
                (id * wsize * hsize + ih * wsize + iw) *
                    csize +
                c;
            for (const auto ic : c10::irange(cend)) {
              auto vals = vec::convert_to_int32<typename T::underlying>(
                  i_p + i_idx + ic * vec_width);
              acc_buffer[ic] = acc_buffer[ic] + vals;
            }
          }
        }
      }
      // convert int32 accumulative to fp32
      vec::convert((int*)acc_buffer, (float*)acc_buffer_fp, cend * vec_width);

      // first quantize using AVX2 or AVX512 using 32 lanes, then 8, finally falls
      // back to single
#ifdef CPU_CAPABILITY_AVX2
      QuantizeAvx2<typename T::underlying>(
          (float*)acc_buffer_fp,
          o_p + c,
          cend * vec_width,
          multiplier,
          output_zero_point);
#else
      QuantizeAvx512<typename T::underlying>(
          (float*)acc_buffer_fp,
          o_p + c,
          cend * vec_width,
          multiplier,
          output_zero_point);
#endif
    }
    c_start = csize / vec_width * vec_width;
  }
#endif
}

template <typename T>
void do_avg_pool_on_AVX_n(
    typename T::underlying* i_p,
    typename T::underlying* o_p,
    int64_t& c,
    int64_t channel_size,
    int64_t channel_multiplier,
    int32_t input_zero_point_m_size,
    int32_t output_zero_point,
    float multiplier,
    int64_t dstart,
    int64_t dend,
    int64_t hstart,
    int64_t hend,
    int64_t wstart,
    int64_t wend,
    int64_t stride_C,
    int64_t stride_D,
    int64_t stride_H,
    int64_t stride_W) {
#if (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)) && !defined(_MSC_VER)
  constexpr int vec_width = Vectorized<T>::size() / 4;
#ifdef CPU_CAPABILITY_AVX2
  if (vec_width == 8) {
#else
  if (vec_width == 16) {
#endif
    for (; c + vec_width <= channel_size; c += vec_width) {
      int64_t tcntr = 0;

      Vectorized<int32_t> acc(input_zero_point_m_size);
      for (const auto id : c10::irange(dstart, dend)) {
        for (const auto ih : c10::irange(hstart, hend)) {
          for (const auto iw : c10::irange(wstart, wend)) {
            tcntr = id * stride_D + ih * stride_H + iw * stride_W;
            auto vals = vec::convert_to_int32<typename T::underlying>(
                i_p + tcntr * channel_multiplier + c * stride_C);
            acc = acc + vals;
          }
        }
      }
      int32_t acc_int[vec_width];
      float acc_fp[vec_width];
      acc.store(acc_int);
      vec::convert(acc_int, acc_fp, vec_width);
      at::native::quantize_vec<T>(
          1.0f / multiplier,
          output_zero_point,
          acc_fp,
          reinterpret_cast<T*>(o_p + c),
          vec_width);
    }
  }
#endif
}

template <typename T>
void _qadaptive_avg_pool_kernel(
    const Tensor& qx,
    Tensor& qy,
    int64_t nBatch,
    int64_t sizeC,
    int64_t isizeD,  // Set to 1 for 2d
    int64_t isizeH,
    int64_t isizeW,
    int64_t osizeD,  // Set to 1 for 2d
    int64_t osizeH,
    int64_t osizeW,
    int64_t istrideB,
    int64_t istrideC,
    int64_t istrideD,  // Set to 1 for 2d
    int64_t istrideH,
    int64_t istrideW) {

  T* idata = static_cast<T*>(qx.data_ptr());
  T* odata = static_cast<T*>(qy.data_ptr());

  const float input_scale = qx.q_scale();
  const float output_scale = qy.q_scale();
  const int input_zero_point = qx.q_zero_point();
  const int output_zero_point = qy.q_zero_point();

  at::parallel_for(0, nBatch, 0, [&](int64_t batch_start, int64_t batch_end) {
    for (const auto b : c10::irange(batch_start, batch_end)) {
      auto* i_p = reinterpret_cast<typename T::underlying*>(
          idata + b * istrideB);

      for (const auto od : c10::irange(osizeD)) {
        int istartD = (int)std::floor((float)(od * isizeD) / osizeD);
        int iendD = (int)std::ceil((float)((od + 1) * isizeD) / osizeD);
        int kD = iendD - istartD;
        for (const auto oh : c10::irange(osizeH)) {
          int istartH = (int)std::floor((float)(oh * isizeH) / osizeH);
          int iendH = (int)std::ceil((float)((oh + 1) * isizeH) / osizeH);
          int kH = iendH - istartH;
          for (const auto ow : c10::irange(osizeW)) {
            auto* o_p = reinterpret_cast<typename T::underlying*>(
                odata +
                b * osizeD * osizeH * osizeW * sizeC +
                od * osizeH * osizeW * sizeC +
                oh * osizeW * sizeC +
                ow * sizeC);
            int istartW = (int)std::floor((float)(ow * isizeW) / osizeW);
            int iendW = (int)std::ceil((float)((ow + 1) * isizeW) / osizeW);
            int kW = iendW - istartW;
            int size = kD * kH * kW;
            float multiplier = input_scale / output_scale / size;
            int input_zero_point_m_size = -input_zero_point * size;
            int64_t c = 0;
            // For int8 or uint8quantization, we implicitly use int32 as
            // accumulation Or else, it will go to the slow path
            // TODO: support 16bit, 32bit, and etc.
            auto* internal_i_p = i_p +
                                istartD * istrideD +
                                istartH * istrideH +
                                istartW * istrideW;

            // Note: If AVX is not available, `do_avg_pool_on_AVX_n is a noop.
            //       In that case, the following loop takes over
            // TODO: more vectorization with loop interleaving
            do_avg_pool_on_AVX_n<T>(
                internal_i_p,
                o_p,
                c,
                sizeC,
                1,
                input_zero_point_m_size,
                output_zero_point,
                multiplier,
                0,
                kD,
                0,
                kH,
                0,
                kW,
                istrideC,
                istrideD,
                istrideH,
                istrideW);
            // 1) The following loop handles the remaining channels
            // 2) It also handles the Non-AVX2 path
            for (; c < sizeC; ++c) {
              int32_t acc_int32 = input_zero_point_m_size;
              int64_t tcntr = 0;
              for (const auto id : c10::irange(kD)) {
                for (const auto ih : c10::irange(kH)) {
                  for (const auto iw : c10::irange(kW)) {
                    tcntr = id * istrideD +
                        ih * istrideH +
                        iw * istrideW;
                    auto val = *(internal_i_p + tcntr + c * istrideC);
                    acc_int32 += val;
                  }
                }
              }
              // clamp
              o_p[c] = at::native::quantize_val<T>(1.0f / multiplier,
                                                          output_zero_point,
                                                          acc_int32).val_;
            } // c
          } // oh
        } // ow
      } // od
    }
  });
}

void qadaptive_avg_pool2d_nhwc_kernel(
    const Tensor& qx,
    Tensor& qy,
    int64_t nBatch,
    int64_t sizeC,
    int64_t isizeH,
    int64_t isizeW,
    int64_t osizeH,
    int64_t osizeW,
    int64_t istrideB,
    int64_t istrideC,
    int64_t istrideH,
    int64_t istrideW) {
    AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "adaptive_avg_pool2d_nhwc", [&]() {
        _qadaptive_avg_pool_kernel<scalar_t>(
          qx,
          qy,
          nBatch,
          sizeC,
          /*isizeD=*/1,
          isizeH,
          isizeW,
          /*osizeD=*/1,
          osizeH,
          osizeW,
          istrideB,
          istrideC,
          /*istrideD=*/1,
          istrideH,
          istrideW);
      }
    );
}

void qadaptive_avg_pool3d_ndhwc_kernel(
    const Tensor& qx,
    Tensor& qy,
    int64_t nBatch,
    int64_t sizeC,
    int64_t isizeD,
    int64_t isizeH,
    int64_t isizeW,
    int64_t osizeD,
    int64_t osizeH,
    int64_t osizeW,
    int64_t istrideB,
    int64_t istrideC,
    int64_t istrideD,
    int64_t istrideH,
    int64_t istrideW) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "adaptive_avg_pool3d_ndhwc", [&]() {
    _qadaptive_avg_pool_kernel<scalar_t>(
      qx,
      qy,
      nBatch,
      sizeC,
      isizeD,
      isizeH,
      isizeW,
      osizeD,
      osizeH,
      osizeW,
      istrideB,
      istrideC,
      istrideD,
      istrideH,
      istrideW);
    }
  );
}

template <typename T>
void _qavg_pool_nhwc_kernel(
    const Tensor& qx,
    Tensor& qy,
    int64_t nBatch,
    int64_t nInputPlane,
    int64_t inputWidth,
    int64_t inputHeight,
    int64_t inputDepth,
    int64_t outputWidth,
    int64_t outputHeight,
    int64_t outputDepth,
    int kW,
    int kH,
    int kD,
    int dW,
    int dH,
    int dD,
    int padW,
    int padH,
    int padD,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  T* idata = static_cast<T*>(qx.data_ptr());
  T* odata = static_cast<T*>(qy.data_ptr());
  int strideC = 1;
  int strideW = strideC * nInputPlane;
  int istrideH = strideW * inputWidth;
  int istrideD = istrideH * inputHeight;
  int istrideB = istrideD * inputDepth;

  // lift these operations outside the loop to reduce access overheads
  float input_scale = qx.q_scale();
  float output_scale = qy.q_scale();
  int input_zero_point = qx.q_zero_point();
  int output_zero_point = qy.q_zero_point();
  int64_t divisor_override_factor =
      divisor_override.has_value() ? divisor_override.value() : 0;

  at::parallel_for(0, nBatch * outputDepth * outputHeight * outputWidth, 0, [&](int64_t begin, int64_t end) {
    int64_t b{0}, od{0}, oh{0}, ow{0};
    data_index_init(begin, b, nBatch, od, outputDepth, oh, outputHeight, ow, outputWidth);

    for (const auto i : c10::irange(begin, end)) {
      auto* i_p = reinterpret_cast<typename T::underlying*>(idata + b * istrideB);
      auto* o_p = reinterpret_cast<typename T::underlying*>(odata + i * strideW);
      int dstart = od * dD - padD;
      int hstart = oh * dH - padH;
      int wstart = ow * dW - padW;

      int dend = std::min(dstart + kD, (int)inputDepth + padD);
      int hend = std::min(hstart + kH, (int)inputHeight + padH);
      int wend = std::min(wstart + kW, (int)inputWidth + padW);
      int pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);

      dstart = std::max(dstart, 0);
      hstart = std::max(hstart, 0);
      wstart = std::max(wstart, 0);
      dend = std::min(dend, (int)inputDepth);
      hend = std::min(hend, (int)inputHeight);
      wend = std::min(wend, (int)inputWidth);

      int size = (dend - dstart) * (hend - hstart) * (wend - wstart);
      int divide_size = count_include_pad ? pool_size : size;
      int divide_factor =
          divisor_override_factor ? divisor_override_factor : divide_size;
      float multiplier = input_scale / output_scale  / divide_factor;
      int input_zero_point_m_size = -input_zero_point * size;

      int c_start = 0;

      // For int8 quantization, we implicitly use int32 as accumulation
      // Or else, it will go to the slow path
      // TODO: support 16bit, 32bit, and etc.
      do_avg_pool_nhwc_on_AVX_n<T>(
          i_p,
          o_p,
          c_start,
          input_zero_point_m_size,
          output_zero_point,
          multiplier,
          dstart,
          dend,
          hstart,
          hend,
          wstart,
          wend,
          inputDepth,
          inputHeight,
          inputWidth,
          nInputPlane);

      // 1) The following loop handles the remaining channels
      // 2) It also handles the Non-AVX2 path
      for (const auto c: c10::irange(c_start, nInputPlane)) {
        int32_t acc_int32 = input_zero_point_m_size;
        for (const auto id : c10::irange(dstart, dend)) {
          for (const auto ih : c10::irange(hstart, hend)) {
            for (const auto iw : c10::irange(wstart, wend)) {
              auto val =
                  *(i_p + id * istrideD + ih * istrideH + iw * strideW +
                  c * strideC);
              acc_int32 += val;
            }
          }
       }
       double acc_fp = acc_int32 * 1.0;
       // clamp
       o_p[c] = at::native::quantize_val<T>(
           1.0f / multiplier, output_zero_point, acc_fp)
           .val_;
      } // c

      data_index_step(b, nBatch, od, outputDepth, oh, outputHeight, ow, outputWidth);
    }
  });
}

void qavg_pool2d_nhwc_kernel(
    const Tensor& qx,
    Tensor& qy,
    int64_t b,
    int64_t nInputPlane,
    int64_t inputWidth,
    int64_t inputHeight,
    int64_t outputWidth,
    int64_t outputHeight,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "avg_pool2d_nhwc", [&]() {
    _qavg_pool_nhwc_kernel<scalar_t>(
      qx,
      qy,
      b,
      nInputPlane,
      inputWidth,
      inputHeight,
      1,
      outputWidth,
      outputHeight,
      1,
      kW,
      kH,
      1,
      dW,
      dH,
      1,
      padW,
      padH,
      0,
      count_include_pad,
      divisor_override);
  });
}

void qavg_pool3d_nhwc_kernel(
    const Tensor& qx,
    Tensor& qy,
    int64_t b,
    int64_t nInputPlane,
    int64_t inputWidth,
    int64_t inputHeight,
    int64_t inputDepth,
    int64_t outputWidth,
    int64_t outputHeight,
    int64_t outputDepth,
    int kW,
    int kH,
    int kD,
    int dW,
    int dH,
    int dD,
    int padW,
    int padH,
    int padD,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "avg_pool3d_nhwc", [&]() {
    _qavg_pool_nhwc_kernel<scalar_t>(
      qx,
      qy,
      b,
      nInputPlane,
      inputWidth,
      inputHeight,
      inputDepth,
      outputWidth,
      outputHeight,
      outputDepth,
      kW,
      kH,
      kD,
      dW,
      dH,
      dD,
      padW,
      padH,
      padD,
      count_include_pad,
      divisor_override);
  });
}

template <typename T>
int64_t do_quantized_bilinear_on_AVX_n(
    const typename T::underlying*& pos1,
    typename T::underlying*& pos2,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t channels,
    int32_t output_zero_point,
    int32_t input_zero_point,
    float inverse_scale,
    const float h0lambda,
    const float h1lambda,
    const float w0lambda,
    const float w1lambda,
    const int64_t h1p,
    const int64_t w1p) {
  int64_t c = 0;
#if (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)) && !defined(_MSC_VER)
  constexpr auto vec_width = Vectorized<T>::size() / 4;
#ifdef CPU_CAPABILITY_AVX2
  if (vec_width == 8) {
#else
  if (vec_width == 16) {
#endif
    for (; c + vec_width <= channels; c += vec_width) {
      Vectorized<float> pos1_fp_v[4];
      Vectorized<int32_t> pos1_int_v[4];
      pos1_int_v[0] = vec::convert_to_int32<typename T::underlying>(pos1);
      pos1_int_v[1] = vec::convert_to_int32<typename T::underlying>(
          pos1 + w1p * channels);
      pos1_int_v[2] = vec::convert_to_int32<typename T::underlying>(
          pos1 + h1p * input_width * channels);
      pos1_int_v[3] = vec::convert_to_int32<typename T::underlying>(
          pos1 + (h1p * input_width + w1p) * channels);
      for (const auto i : c10::irange(4)) {
        int32_t pos1_int[vec_width];
        float pos1_fp[vec_width];
        pos1_int_v[i].store(pos1_int);
        vec::convert(pos1_int, pos1_fp, vec_width);
        pos1_fp_v[i] = Vectorized<float>::loadu(pos1_fp, 8);
      }
      Vectorized<float> h0lambda_v(h0lambda);
      Vectorized<float> h1lambda_v(h1lambda);
      Vectorized<float> w0lambda_v(w0lambda);
      Vectorized<float> w1lambda_v(w1lambda);
      Vectorized<float> input_zero_point_v(input_zero_point);
      Vectorized<float> result =
          h0lambda_v * (w0lambda_v * pos1_fp_v[0] + w1lambda_v * pos1_fp_v[1]) +
          h1lambda_v * (w0lambda_v * pos1_fp_v[2] + w1lambda_v * pos1_fp_v[3]) -
          input_zero_point_v;
      float result_fp[vec_width];
      result.store(result_fp);
      at::native::quantize_vec<T>(
          inverse_scale,
          output_zero_point,
          result_fp,
          reinterpret_cast<T*>(pos2),
          vec_width);
      pos1 += vec_width;
      pos2 += vec_width;
    }
  }
#endif
  return c;
}

void qupsample_bilinear2d_nhwc_kernel(
    Tensor& output,
    const Tensor& input,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  AT_DISPATCH_QINT_TYPES(input.scalar_type(), "upsample_bilinear2d_nhwc", [&]() {
    auto* idata = static_cast<scalar_t*>(input.data_ptr());
    auto* odata = static_cast<scalar_t*>(output.data_ptr());
    float inverse_scale = output.q_scale() / input.q_scale();
    const auto rheight = area_pixel_compute_scale<float>(
        input_height, output_height, align_corners, scales_h);
    const auto rwidth = area_pixel_compute_scale<float>(
        input_width, output_width, align_corners, scales_w);

    auto input_q_zero_point = input.q_zero_point();
    auto output_q_zero_point = output.q_zero_point();
    at::parallel_for(0, nbatch * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
      int64_t b{0}, h2{0}, w2{0};
      data_index_init(begin, b, nbatch, h2, output_height, w2, output_width);

      for ([[maybe_unused]] const auto i : c10::irange(begin, end)) {
        auto* i_p = reinterpret_cast<typename scalar_t::underlying*>(
            idata + b * input_height * input_width * channels);
        auto* o_p = reinterpret_cast<typename scalar_t::underlying*>(
            odata + b * output_height * output_width * channels);

        const auto h1r = area_pixel_compute_source_index<float>(
            rheight, h2, align_corners, /*cubic=*/false);

        const int64_t h1 = h1r;
        const int64_t h1p = (h1 < input_height - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = static_cast<float>(1.) - h1lambda;

        const auto w1r = area_pixel_compute_source_index<float>(
            rwidth, w2, align_corners, /*cubic=*/false);
        const int64_t w1 = w1r;
        const int64_t w1p = (w1 < input_width - 1) ? 1 : 0;

        const float w1lambda = w1r - w1;
        const float w0lambda = static_cast<float>(1.) - w1lambda;

        int64_t c = 0;
        // We use float32 to do the computation
        const typename scalar_t::underlying* pos1 =
            i_p + (h1 * input_width + w1) * channels;
        typename scalar_t::underlying* pos2 =
            o_p + (h2 * output_width + w2) * channels;
        // We have to isolate this function out because the VS does not
        // expand the macro correctly.
        c = do_quantized_bilinear_on_AVX_n<scalar_t>(
            pos1,
            pos2,
            input_width,
            output_height,
            output_width,
            channels,
            output_q_zero_point,
            input_q_zero_point,
            inverse_scale,
            h0lambda,
            h1lambda,
            w0lambda,
            w1lambda,
            h1p,
            w1p);
        // 1) The following loop handles the remaining channels
        // 2) It also handles the Non-AVX2 path
        for (; c < channels; ++c) {
          float result = h0lambda *
                  (w0lambda * pos1[0] + w1lambda * pos1[w1p * channels]) +
              h1lambda *
                  (w0lambda * pos1[h1p * input_width * channels] +
                   w1lambda * pos1[(h1p * input_width + w1p) * channels]);
          pos2[0] = at::native::quantize_val<scalar_t>(
                        inverse_scale,
                        output_q_zero_point,
                        result - input_q_zero_point)
                        .val_;
          pos1 += 1;
          pos2 += 1;
        } // c

        data_index_step(b, nbatch, h2, output_height, w2, output_width);
      }
    });
  });
}

void qtopk_kernel(Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  auto sizes = self.sizes();
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(sizes, /*squash_dims=*/dim)
    .add_output(values)
    .add_output(indices)
    .add_input(self)
    .build();

  auto mode_values_stride = values.strides()[dim];
  auto mode_indices_stride = indices.strides()[dim];
  auto tmp_values_stride = self.strides()[dim];
  // If sizes is empty, the tensor is scalar. This prevents accessing an empty array.
  auto dim_size = sizes.empty() ? 1 : sizes[dim];

  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "qtopk_cpu", [&] {
    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
      using underlying_t = typename scalar_t::underlying;
      static_assert(sizeof(scalar_t) == sizeof(underlying_t), "");
      return topk_impl_loop<underlying_t, underlying_t>(
          mode_values_stride, mode_indices_stride, tmp_values_stride,
          k, dim_size, largest, sorted, data, strides, n);
    };

    int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, sizes[dim]);
    iter.for_each(loop, /*grain_size=*/grain_size);
  });
}

template <typename T, bool ReluFused>
inline void do_bn_compute(
    typename T::underlying* X_ptr,
    typename T::underlying* Y_ptr,
    Vectorized<float> & fake_scale,
    Vectorized<float> & in_zp_vec,
    Vectorized<float> & scale_neg_zp_premul,
    int64_t out_zero_point,
    Vectorized<T> & out_zero_point_v,
    float*  alpha,
    float* beta,
    int64_t vec_num,
    int64_t kVLen
) {
  using Vec = Vectorized<T>;
  auto vals_q = Vec::loadu(X_ptr);
  // Fake scale of 1.0 here, should not affect performance (FMA in place of sub)
  auto vals_dq = vals_q.dequantize(fake_scale, in_zp_vec, scale_neg_zp_premul);
  for (const auto idx : c10::irange(vec_num)) {
    auto alpha_v = Vectorized<float>::loadu(alpha + idx * kVLen);
    auto beta_v = Vectorized<float>::loadu(beta + idx * kVLen);
    vals_dq[idx] = vec::fmadd(alpha_v, vals_dq[idx], beta_v);
  }
  auto outputs_q = Vec::quantize(vals_dq, /*scale=*/1.0f, out_zero_point, /*inverse_scale=*/1.0f);
  // Fake scale again
  if constexpr (ReluFused) {
    outputs_q = outputs_q.maximum(out_zero_point_v);
  }
  outputs_q.store(Y_ptr, vec_num * kVLen);
}

template <bool ReluFused>
void q_batch_norm_kernel(
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t in_zero_point,
    int64_t out_zero_point,
    const Tensor& input,
    const Tensor& a,
    const Tensor& b,
    Tensor& output) {

  AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qbatch_norm", [&]() {
    float* alpha = a.data_ptr<float>();
    float* beta = b.data_ptr<float>();
    auto minimum = std::numeric_limits<scalar_t::underlying>::lowest();
    auto maximum = std::numeric_limits<scalar_t::underlying>::max();
    scalar_t::underlying* X =
        reinterpret_cast<scalar_t::underlying*>(input.data_ptr());
    scalar_t::underlying* Y = reinterpret_cast<scalar_t::underlying*>(output.data_ptr());

    constexpr int kVLen = Vectorized<float>::size();
    const int64_t outer_size = N * HxW;
    using Vec = Vectorized<scalar_t>;
    // Hoisted variables
    auto in_zp_vec = Vectorized<float>(static_cast<float>(in_zero_point));
    auto fake_scale = Vectorized<float>(1.0f);
    auto scale_neg_zp_premul = fake_scale * in_zp_vec.neg();
    auto out_zero_point_v = Vec(scalar_t(out_zero_point));
    const auto lanes = static_cast<int64_t>(Vec::float_num_vecs() * kVLen);
    at::parallel_for(0, outer_size, 0, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        auto* X_ptr = reinterpret_cast<typename scalar_t::underlying*>(X + i * C);
        auto* Y_ptr = reinterpret_cast<typename scalar_t::underlying*>(Y + i * C);
        int64_t ch = 0;

        for(; ch + lanes <= C; ch += lanes) {
          do_bn_compute<scalar_t, ReluFused>(
            X_ptr + ch,
            Y_ptr + ch,
            fake_scale,
            in_zp_vec,
            scale_neg_zp_premul,
            out_zero_point,
            out_zero_point_v,
            alpha + ch,
            beta + ch,
            Vec::float_num_vecs(),
            kVLen
          );
        }

        // for channel between 8 and 32, still use 32 width for performance
        // Benchmark shows it is faster than doing 8 channels each time
        int64_t elem_size = C - ch;
        if ((lanes == 32) && elem_size >= kVLen) {
          int64_t vec_num = elem_size / kVLen;
          std::vector<typename scalar_t::underlying> buf_in(lanes);
          memcpy(buf_in.data(), X_ptr + ch, vec_num * kVLen); // 3 cycles
          do_bn_compute<scalar_t, ReluFused>(
            buf_in.data(),
            Y_ptr + ch,
            fake_scale,
            in_zp_vec,
            scale_neg_zp_premul,
            out_zero_point,
            out_zero_point_v,
            alpha + ch,
            beta + ch,
            vec_num,
            kVLen
          );
          ch += vec_num * kVLen;
        }
        // for channels less than 8
        for (; ch < C; ++ch) {
          long quantized_down = out_zero_point +
              lrintf(alpha[ch] * (X_ptr[ch] - in_zero_point) +
                          beta[ch]);
          if constexpr (ReluFused) { // static if
            quantized_down = std::max<long>(quantized_down, out_zero_point);
          }
          Y_ptr[ch] = std::min<long>(
              std::max<long>(quantized_down, minimum), maximum);
        }
      }
    });
  });
}

template <typename T>
void q_batch_norm_cpu_kernel_impl(
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t in_zero_point,
    int64_t out_zero_point,
    const uint8_t* in_ptr,
    const float* alpha_ptr,
    const float* beta_ptr,
    T* out_ptr) {

  int q_min = 0;
  int q_max = 255;
  const int64_t outer_size = N * HxW;

#if defined(CPU_CAPABILITY_AVX512)
  constexpr int kVLen = 16;
  static constexpr int num_vecs = sizeof(float) / sizeof(uint8_t);
  auto in_zp_vec = _mm512_set1_ps((float)in_zero_point);
  auto fake_scale = _mm512_set1_ps(1.0f);
  auto scale_neg_zp_premul = _mm512_xor_ps(_mm512_set1_ps(-0.f), in_zp_vec);
  auto out_zero_point_v = _mm512_set1_epi32((int)out_zero_point);
  constexpr auto lanes = static_cast<int64_t>(num_vecs * kVLen);
  __m512i v_q_max = _mm512_set1_epi32(q_max);
  __m512i v_q_min = _mm512_set1_epi32(q_min);

  auto load_convert_u8_to_f32_512bit = [&](const uint8_t* src, __m512* dst) {
    // Step 1: Load 512 bits
    __m512i raw = _mm512_loadu_si512(src);

    // Step 2: Extract two 256-bit chunks
    __m256i v0 = _mm512_extracti64x4_epi64(raw, 0); // bytes 0–31
    __m256i v1 = _mm512_extracti64x4_epi64(raw, 1); // bytes 32–63

    // Step 3: Process each 256-bit chunk
    // --- Expand uint8_t -> uint16_t ---
    __m256i u16lo0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v0, 0));
    __m256i u16hi0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v0, 1));
    __m256i u16lo1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v1, 0));
    __m256i u16hi1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v1, 1));
    // --- Expand to uint32_t and convert to float ---
    dst[0] = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(u16lo0));
    dst[1] = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(u16hi0));
    dst[2] = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(u16lo1));
    dst[3] = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(u16hi1));
  };

  auto load_convert_u8_to_f32_128bit = [&](const uint8_t* src) {
    // --- Load and expand uint8_t -> uint16_t ---
    __m256i v_u16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)src));
    // --- Expand to uint32_t and convert to float ---
    return _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(v_u16));
  };

  auto store_output = [&](__m512 out, T* out_addr) {
    if constexpr (std::is_same<T, float>::value) {
      _mm512_storeu_ps(out_addr, out);
    } else if constexpr (std::is_same<T, at::BFloat16>::value) {
      __m256i out_bf16 = cvtfp32_bf16(out);
      _mm256_storeu_si256((__m256i*)out_addr, out_bf16);
    } else if constexpr (std::is_same<T, at::Half>::value) {
      __m256i out_f16 = cvtfp32_fp16(out);
      _mm256_storeu_si256((__m256i*)out_addr, out_f16);
    } else { //  T == uint8, requantization needed
      __m512i out_i32 = _mm512_cvtps_epi32(out);
      out_i32 = _mm512_add_epi32(out_i32, out_zero_point_v);
      out_i32 = _mm512_min_epi32(out_i32, v_q_max);
      out_i32 = _mm512_max_epi32(out_i32, v_q_min);
      __m128i out_i8 = _mm512_cvtepi32_epi8(out_i32);
      _mm_storeu_si128((__m128i*)out_addr, out_i8);
    }
  };
#endif

  at::parallel_for(0, outer_size, 0, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      auto* X_ptr = in_ptr + i * C;
      auto* Y_ptr = out_ptr + i * C;
      int64_t ch = 0;

#if defined(CPU_CAPABILITY_AVX512)
      __m512 vals_dq[num_vecs];
      for(; ch + lanes <= C; ch += lanes) {
        // load 64 values of input then dequantize them
        load_convert_u8_to_f32_512bit(X_ptr + ch, vals_dq);
        for (const auto idx : c10::irange(num_vecs)) {
          vals_dq[idx] = _mm512_fmadd_ps(fake_scale, vals_dq[idx], scale_neg_zp_premul);
          auto alpha_v = _mm512_loadu_ps(alpha_ptr + ch + idx * kVLen);
          auto beta_v = _mm512_loadu_ps(beta_ptr + ch + idx * kVLen);
          vals_dq[idx] = _mm512_fmadd_ps(alpha_v, vals_dq[idx], beta_v);
          store_output(vals_dq[idx], Y_ptr + ch + idx * kVLen);
        }
      }

      // for channel between 16 and 64
      int64_t elem_size = C - ch;
      if (elem_size >= kVLen) {
        int64_t vec_num = elem_size / kVLen;
        for (const auto idx : c10::irange(vec_num)) {
          __m512 val_dq = load_convert_u8_to_f32_128bit(X_ptr + ch + idx * kVLen);
          val_dq = _mm512_fmadd_ps(fake_scale, val_dq, scale_neg_zp_premul);
          auto alpha_v = _mm512_loadu_ps(alpha_ptr + ch + idx * kVLen);
          auto beta_v = _mm512_loadu_ps(beta_ptr + ch + idx * kVLen);
          val_dq = _mm512_fmadd_ps(alpha_v, val_dq, beta_v);
          store_output(val_dq, Y_ptr + ch + idx * kVLen);
        }
        ch += vec_num * kVLen;
      }
#endif
      // for channels less than 16
      for (; ch < C; ++ch) {
        float y_val_f = alpha_ptr[ch] * (X_ptr[ch] - in_zero_point) +
                        beta_ptr[ch];
        if constexpr (std::is_same<T, float>::value) {
          Y_ptr[ch] = y_val_f;
        } else if constexpr (std::is_same<T, at::BFloat16>::value) {
          Y_ptr[ch] = (at::BFloat16)y_val_f;
        } else if constexpr (std::is_same<T, at::Half>::value) {
          Y_ptr[ch] = (at::Half)y_val_f;
        } else { //  T == uint8, requantization needed
          long quantized_down = out_zero_point + lrintf(y_val_f);
          Y_ptr[ch] = std::min<long>(
              std::max<long>(quantized_down, q_min), q_max);
        }
      }
    }
  });
}

void q_batch_norm_cpu_kernel(
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t in_zero_point,
    int64_t out_zero_point,
    const Tensor& input,
    const Tensor& a,
    const Tensor& b,
    Tensor& output) {
  auto in_ptr = input.const_data_ptr<uint8_t>();
  float* alpha_ptr = a.data_ptr<float>();
  float* beta_ptr = b.data_ptr<float>();
  AT_DISPATCH_FLOATING_TYPES_AND3(
      at::ScalarType::BFloat16, at::ScalarType::Half, at::ScalarType::Byte, output.scalar_type(), "int8_batch_norm2d_cpu", [&] {
        auto out_ptr = output.data_ptr<scalar_t>();
        q_batch_norm_cpu_kernel_impl<scalar_t>(
            N, C, HxW, in_zero_point, out_zero_point, in_ptr, alpha_ptr, beta_ptr, out_ptr);
      });
}

void _fake_quantize_tensor_helper(
  Tensor& output,
  Tensor& mask,
  const Tensor& input,
  int fake_quant_on,
  float sc,
  int64_t z_point,
  int64_t quant_min,
  int64_t quant_max) {

  float inv_scale = 1.0f / sc;

  auto iter_combined = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(output)
    .add_output(mask)
    .add_input(input)
    .build();

  if (at::isReducedFloatingType(input.scalar_type())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "fake_quantize_tensor_cachemask_kernel_type_handling", [&]() {
      iter_combined.for_each([&](char** data, const int64_t* strides, int64_t n) {
        for (const auto i : c10::irange(n)) {
          scalar_t* output_val = (scalar_t*)(data[0] + i * strides[0]);
          bool* mask_val = (bool*)(data[1] + i * strides[1]);
          scalar_t* input_val = (scalar_t*)(data[2] + i * strides[2]);

          if (fake_quant_on) {
            auto qval_f = z_point + std::nearbyint(*input_val * inv_scale);
            const auto qval = static_cast<int64_t>(std::fmin(std::fmax(qval_f, quant_min), quant_max));
            *output_val = (qval - z_point) * sc;
            *mask_val = ((quant_min <= qval_f) && (qval_f <= quant_max));
          } else {
            *output_val = *input_val;
            *mask_val = 1;
          }
        }
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fake_quantize_tensor_cachemask_kernel_type_handling", [&] {
      iter_combined.for_each([&](char** data, const int64_t* strides, int64_t n) {
        for (const auto i : c10::irange(n)) {
          scalar_t* output_val = (scalar_t*)(data[0] + i * strides[0]);
          bool* mask_val = (bool*)(data[1] + i * strides[1]);
          scalar_t* input_val = (scalar_t*)(data[2] + i * strides[2]);

          if (fake_quant_on) {
            auto qval_f = z_point + std::nearbyint(*input_val * inv_scale);
            const auto qval = static_cast<int64_t>(std::fmin(std::fmax(qval_f, quant_min), quant_max));
            *output_val = (qval - z_point) * sc;
            *mask_val = ((quant_min <= qval_f) && (qval_f <= quant_max));
          } else {
            *output_val = *input_val;
            *mask_val = 1;
          }
        }
      });
    });
  }
}

void fake_quantize_tensor_cachemask_kernel(
    Tensor& output,
    Tensor& mask,
    const Tensor& input,
    float sc,
    int64_t z_point,
    int64_t quant_min,
    int64_t quant_max) {
  _fake_quantize_tensor_helper(output, mask, input, 1, sc, z_point, quant_min, quant_max);
}

void fake_quantize_tensor_cachemask_tensor_qparams_kernel(
    Tensor& output,
    Tensor& mask,
    const Tensor& input,
    const Tensor& sc,
    const Tensor& z_point,
    const Tensor& fake_quant_enabled,
    int64_t quant_min,
    int64_t quant_max) {
  _fake_quantize_tensor_helper(output, mask, input, fake_quant_enabled.item().toInt(), sc.item().toFloat(), z_point.item().toInt(), quant_min, quant_max);
}

void fake_quantize_learnable_tensor_grad_kernel_cpu(
    TensorIterator& iter,
    float scale,
    float inv_scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    float grad_factor) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  float dscale_small = quant_min - zero_point;
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  float dscale_big = quant_max - zero_point;
  iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
    /*  When a for_each call is made on a TensorIterator with multiple inputs and outputs,
        the order they are accessed follows the order they are built within the iterator.
        For example, if an iterator is built in the following order:
        auto iter = TensorIteratorConfig().
          .add_output(firstOutput)
          .add_output(secondOutput)
          .add_input(firstInput)
          .add_input(secondInput)
          .build()
        data will contain 4 pointers to pointers to values in the following order:
        firstOutput, secondOutput, firstInput, secondInput.
        Proper pointer referencing and dereferencing, along with the usage of strides
        (to move onto different elements), can allow accessing of the input and assignment
        to the right output.
    */
    for (const auto i : c10::irange(n)) {
      float* dXOutput = (float*)(data[0] + i * strides[0]);
      float* dScaleOutput = (float*)(data[1] + i * strides[1]);
      float* dZeroPointOutput = (float*)(data[2] + i * strides[2]);
      float* XInput = (float*)(data[3] + i * strides[3]);
      float* dYInput = (float*)(data[4] + i * strides[4]);
      // Calculate gradients for X.
      int64_t xqi = std::nearbyint(zero_point + (*XInput) * inv_scale);
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      *dXOutput = (*dYInput) * (xqi >= quant_min && xqi <= quant_max);
      // Calculate gradients for scale and zero point.
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      float xfqi = static_cast<float>((std::max(std::min(xqi, quant_max), quant_min) - zero_point) * scale);
      // Calculate gradients according to the gradient of the clamp function.
      if (xqi < quant_min || xqi > quant_max) {
        *dZeroPointOutput = (*dYInput) * (-1) * scale * grad_factor;
        *dScaleOutput = ((xqi < quant_min) ? ((*dYInput) * dscale_small) : ((*dYInput) * dscale_big)) * grad_factor;
      } else {
        *dZeroPointOutput = 0;
        *dScaleOutput = (*dYInput) * (xfqi - (*XInput)) * inv_scale * grad_factor;
      }
    }
  });
}

template <typename SelfType>
void _fake_quant_per_channel_cachemask_cpu_helper(
    TensorIterator& iter,
    TensorIterator& iter_mask,
    const int64_t quant_min,
    const int64_t quant_max) {

  const auto& zero_point_dtype = iter.input_dtype(2);

  if(at::isFloatingType(zero_point_dtype)){
    // When zero_point is float, quantize mirroring affine quantizer equation
    // Xq = Round(Xf * inv_scale + zero_point)
    // where zero_point is in float.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(zero_point_dtype, "fake_quantize_channel_cachemask_cpu_zero_point_handling", [&] {
      // write mask
      cpu_kernel(iter_mask, [=](SelfType self, float scale, scalar_t zero_point) -> bool {
        float inv_scale = 1.0f / scale;
        const auto qval = std::lrintf(zero_point + (self * inv_scale));
        return ((quant_min <= qval) && (qval <= quant_max));
      });

      // write fake_quant
      cpu_kernel(iter, [=](SelfType self, float scale, scalar_t zero_point) -> SelfType {
        float inv_scale = 1.0f / scale;
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        return (std::fmin(
                    std::fmax(
                        std::lrintf(zero_point + self * inv_scale),
                        quant_min),
                    quant_max) -
                zero_point) *
            scale;
      });
    });

  } else {
      // write mask
      cpu_kernel(iter_mask, [=](SelfType self, float scale, int32_t zero_point) -> bool {
        float inv_scale = 1.0f / scale;
        const auto qval = static_cast<int64_t>(zero_point + std::nearbyint(self * inv_scale));
        return ((quant_min <= qval) && (qval <= quant_max));
      });

      // write fake_quant
      cpu_kernel(iter, [=](SelfType self, float scale, int32_t zero_point) -> SelfType {
        float inv_scale = 1.0f / scale;
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        return (std::fmin(
                    std::fmax(
                        static_cast<int64_t>(
                            zero_point + std::nearbyint(self * inv_scale)),
                        quant_min),
                    quant_max) -
                zero_point) *
            scale;
      });
  }

}


void fake_quant_per_channel_cachemask_cpu(
    TensorIterator& iter,
    TensorIterator& iter_mask,
    int64_t quant_min,
    int64_t quant_max) {
  // TODO(future, optional): read once, write twice.  Not done at the moment
  //   for simplicity, as we do not expect this to be a bottleneck.

  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "fake_quantize_channel_cachemask_cpu_type_handling", [&]() {
      _fake_quant_per_channel_cachemask_cpu_helper<scalar_t>(iter, iter_mask, quant_min, quant_max);
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "fake_quantize_channel_cachemask_cpu_type_handling", [&] {
      _fake_quant_per_channel_cachemask_cpu_helper<scalar_t>(iter, iter_mask, quant_min, quant_max);
    });
  }
}


void fake_quantize_learnable_channel_grad_kernel_cpu(
    TensorIterator& iter,
    int64_t quant_min,
    int64_t quant_max,
    float grad_factor) {
  iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
    /*  To see how the input and outputs are referenced and assigned,
        please see the implementation of
        fake_quantize_learnable_tensor_grad_kernel_cpu.
    */
    for (const auto i : c10::irange(n)) {
      float* dx_output = (float*)(data[0] + i * strides[0]);
      float* dscale_output = (float*)(data[1] + i * strides[1]);
      float* dzero_point_output = (float*)(data[2] + i * strides[2]);
      float* x_input = (float*)(data[3] + i * strides[3]);
      float* dy_input = (float*)(data[4] + i * strides[4]);
      float* scale_input = (float*)(data[5] + i * strides[5]);
      float* zero_point_input = (float*)(data[6] + i * strides[6]);

      float inv_scale = 1.0f / (*scale_input);
      float dscale_small = quant_min - (*zero_point_input);
      float dscale_big = quant_max - (*zero_point_input);
      // Calculate gradients for X.
      int64_t xqi = std::nearbyint((*zero_point_input) + (*x_input) * inv_scale);
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      *dx_output = (*dy_input) * (xqi >= quant_min && xqi <= quant_max);
      // Calculate gradients for scale and zero point.
      float xfqi = ((std::max(std::min(xqi, quant_max), quant_min) - (*zero_point_input)) * (*scale_input));
      if (xqi < quant_min || xqi > quant_max) {
        *dzero_point_output = (*dy_input) * (-1) * (*scale_input) * grad_factor;
        *dscale_output = ((xqi < quant_min) ? ((*dy_input) * dscale_small) : ((*dy_input) * dscale_big)) * grad_factor;
      } else {
        *dzero_point_output = 0;
        *dscale_output = (*dy_input) * (xfqi - (*x_input)) * inv_scale * grad_factor;
      }
    }
  });
}

// Assumes X is composed of M groups of N elements. Normalizes each of the
// groups and optionally applies affine scaling. Useful for LayerNorm,
// GroupNorm, InstanceNorm.
void quantized_normalize_kernel(
    const Tensor& X, // input tensor
    const Tensor& gamma, // weight (optional)
    const Tensor& beta, // bias (optional)
    bool affine_per_channel, // scaling applied elementwise if false, per channel if true
    int num_channels, // only used if affine_per_channel is set
    int num_groups, // only used if affine_per_channel is set
    int64_t M, // number of groups
    int64_t N, // number of elements in each group
    double eps,
    Tensor* Y) {
  AT_DISPATCH_QINT_TYPES(X.scalar_type(), "quantized_layer_norm_kernel_impl_cpu", [&]() {
    using qVec = vec::Vectorized<scalar_t>;
    using fVec = vec::Vectorized<float>;

    TORCH_INTERNAL_ASSERT(X.numel() == M * N, "Unexpected num elements in X");
    TORCH_INTERNAL_ASSERT(
        !gamma.defined() ||
        (!affine_per_channel && gamma.numel() == N) ||
        (affine_per_channel && gamma.numel() == num_channels),
        "Unexpected size of gamma");
    TORCH_INTERNAL_ASSERT(
        !beta.defined() ||
        (!affine_per_channel && beta.numel() == N) ||
        (affine_per_channel && beta.numel() == num_channels),
        "Unexpected size of beta");

    scalar_t* X_data = X.data_ptr<scalar_t>();
    const float* gamma_data = gamma.defined() ? gamma.const_data_ptr<float>() : nullptr;
    const float* beta_data = beta.defined() ? beta.const_data_ptr<float>() : nullptr;
    scalar_t* Y_data = Y->data_ptr<scalar_t>();
    const bool gamma_null = gamma_data == nullptr;
    const bool beta_null = beta_data == nullptr;
    int64_t x_zp = X.q_zero_point();
    float x_scale = X.q_scale();
    fVec x_zp_vec(x_zp);
    fVec one_vec(1.0f);
    fVec zero_vec(0.0f);
    float x_fake_scale = 1.0f;
    fVec x_fake_scale_vec(x_fake_scale);
    fVec x_fake_scale_zp_neg_premul_vec = x_fake_scale_vec * x_zp_vec.neg();
    int64_t y_zp = Y->q_zero_point();
    float y_scale = Y->q_scale();
    float y_inv_scale = 1.0f / y_scale;

    constexpr int kFloatVLen = fVec::size();
    int64_t kIntVLen = kFloatVLen * qVec::float_num_vecs();
    int64_t kNumIntVecInLayer = N / kIntVLen;
    int64_t kNonVecRemInLayer = N % kIntVLen;
    int channels_per_group = num_channels / num_groups;
    int64_t NPerChannel = N / channels_per_group;
    int64_t kNumIntVecInChannel = NPerChannel / kIntVLen;
    int64_t kNonVecRemInChannel = NPerChannel % kIntVLen;

    at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
      for (const auto i : c10::irange(start, end)) {

        scalar_t* X_ptr = X_data + i * N;
        scalar_t* Y_ptr = Y_data + i * N;

        // First pass: calculate mean and variance.

        scalar_t::underlying* X_ptr_underlying = reinterpret_cast<scalar_t::underlying*>(X_ptr);
        auto l_sum_shifted = hsum(X_ptr_underlying, N);
        auto l_sum_sq_shifted = hsum_sq(X_ptr_underlying, N);
        float l_mean_shifted_div_scale_x = static_cast<float>(l_sum_shifted) / N;
        // mean(dqX) / scale_x
        float layer_mean_div_scale_x = l_mean_shifted_div_scale_x - x_zp;
        // var(dqX) / scale_x^2
        float layer_var_div_scale_x_sq =
          std::max(static_cast<float>(l_sum_sq_shifted) / N -
              l_mean_shifted_div_scale_x * l_mean_shifted_div_scale_x, 0.0f);
        // scale_x / sqrt(var(dqX) + eps)
        float scale_x_div_layer_std = x_scale /
          std::sqrt(layer_var_div_scale_x_sq * x_scale * x_scale + eps);
        fVec layer_mean_div_scale_xVec(layer_mean_div_scale_x);
        fVec scale_x_div_layer_stdVec(scale_x_div_layer_std);

        // Second pass: normalize

        // TODO replace with TensorIterator implementation once #33166 is fixed.
        if (affine_per_channel) {

          // if scaling per channel, scaling parameters can be pre-multiplied
          // with normalization parameters
          for (const auto chIdx : c10::irange(channels_per_group)) {
            int scalingIdx = (i * channels_per_group + chIdx) % (num_channels);
            float gamma = gamma_null ? 1.0f : gamma_data[scalingIdx];
            // scale_x / layer_std * gamma
            float gamma_p = scale_x_div_layer_std * gamma;
            float beta = beta_null ? 0.0f : beta_data[scalingIdx];
            fVec gamma_p_vec(gamma_p);
            fVec beta_vec(beta);

            int64_t chStartIdx = chIdx * NPerChannel;
            int64_t chEndIdx = chStartIdx + NPerChannel;

            for (const auto vecIdx : c10::irange(kNumIntVecInChannel)) {
              int64_t vecStartIdx = chStartIdx + vecIdx * kIntVLen;
              auto qXVec = qVec::loadu(X_ptr + vecStartIdx);
              auto dqXVec = qXVec.dequantize(x_fake_scale_vec, x_zp_vec,
                  x_fake_scale_zp_neg_premul_vec);
              for (auto &dq : dqXVec) {
                dq =
                  (dq - layer_mean_div_scale_xVec) *
                    gamma_p_vec + beta_vec;
              }
              qVec::quantize(dqXVec, y_scale, y_zp, y_inv_scale)
                .store(Y_ptr + vecStartIdx);
            }

            // Remainder
            if (kNonVecRemInChannel > 0) {
              int64_t remIdx = chEndIdx - kNonVecRemInChannel;
              auto qXVec = qVec::loadu(X_ptr + remIdx, kNonVecRemInChannel);
              auto dqXVec = qXVec.dequantize(x_fake_scale_vec, x_zp_vec,
                    x_fake_scale_zp_neg_premul_vec);
              int validDqvecLen = (kNonVecRemInChannel - 1) / fVec::size() + 1;
              for (int i = 0; i < validDqvecLen; ++i) {
                auto &dq = dqXVec[i];
                dq =
                  (dq - layer_mean_div_scale_xVec) *
                    gamma_p_vec + beta_vec;
              }
              qVec::quantize(dqXVec, y_scale, y_zp, y_inv_scale)
                .store(Y_ptr + remIdx, kNonVecRemInChannel);
            }
          } // chIdx

        } else {

          for (const auto vecIdx : c10::irange(kNumIntVecInLayer)) {
            int64_t vecStartIdx = vecIdx * kIntVLen;
            auto qXVec = qVec::loadu(X_ptr + vecStartIdx);
            auto dqXVec = qXVec.dequantize(x_fake_scale_vec, x_zp_vec,
                x_fake_scale_zp_neg_premul_vec);
            for (const auto dqXVecIdx : c10::irange(dqXVec.size())) {
              int64_t vecVecStartIdx = vecStartIdx + dqXVecIdx * kFloatVLen;
              auto gammaVec = gamma_null
                ? one_vec
                : fVec::loadu(gamma_data + vecVecStartIdx);
              auto betaVec = beta_null
                ? zero_vec
                : fVec::loadu(beta_data + vecVecStartIdx);
              dqXVec[dqXVecIdx] =
                (dqXVec[dqXVecIdx] - layer_mean_div_scale_xVec) *
                  scale_x_div_layer_stdVec * gammaVec + betaVec;
              qVec::quantize(dqXVec, y_scale, y_zp, y_inv_scale)
                .store(Y_ptr + vecStartIdx);
            }
          }
          for (int64_t remIdx = N - kNonVecRemInLayer; remIdx < N; remIdx++) {
            const float gamma_v = gamma_null ? 1.0f : gamma_data[remIdx];
            const float beta_v = beta_null ? 0.0f : beta_data[remIdx];
            auto qXVal = X_ptr[remIdx];
            float dqXVal = at::native::dequantize_val(x_fake_scale, x_zp, qXVal);
            float dqY =
              ((dqXVal - layer_mean_div_scale_x) * scale_x_div_layer_std) * gamma_v + beta_v;
            Y_ptr[remIdx] = at::native::quantize_val<scalar_t>(y_scale, y_zp, dqY);
          }
        }
      }
    }); // parallel_for

  });
}

void qmean_inner_dim_kernel(
    const Tensor& self,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    std::optional<ScalarType> opt_dtype,
    Tensor& result) {
  // 'opt_dtype' should be none or equal to that of input
  ScalarType dtype = self.scalar_type();
  auto in_dims = self.sizes().vec();
  auto out_dims = in_dims;
  bool is_all_reduce = !opt_dim.has_value() || opt_dim.value().empty();
  size_t num_dims_to_squeeze = is_all_reduce ? self.dim() : opt_dim.value().size();
  int64_t M = 1; // Num of groups
  int64_t N = 1; // Num of elements to take average of in each group
  for (size_t i = 0; i < in_dims.size() - num_dims_to_squeeze; ++i) {
    M *= in_dims[i];
  }
  for (size_t i = 0; i < num_dims_to_squeeze; ++i) {
    auto idx = out_dims.size() - 1 - i;
    N *= out_dims[idx];
    out_dims[idx] = 1;
  }
  if (!keepdim) {
    out_dims.erase(out_dims.end() - num_dims_to_squeeze, out_dims.end());
  }
  result = at::_empty_affine_quantized(
      out_dims,
      at::device(kCPU).dtype(dtype).memory_format(self.suggest_memory_format()),
      self.q_scale(),
      self.q_zero_point(),
      std::nullopt);

  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "quantized_mean_kernel_impl_cpu", [&]() {
    scalar_t* X_data = self.data_ptr<scalar_t>();
    scalar_t* Y_data = result.data_ptr<scalar_t>();

    at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
      for (const auto i : c10::irange(start, end)) {
        scalar_t* X_ptr = X_data + i * N;
        scalar_t* Y_ptr = Y_data + i;
        scalar_t::underlying* X_ptr_underlying = reinterpret_cast<scalar_t::underlying*>(X_ptr);
        scalar_t::underlying* Y_ptr_underlying = reinterpret_cast<scalar_t::underlying*>(Y_ptr);
        auto x_sum = hsum(X_ptr_underlying, N);
        float y_float = static_cast<float>(x_sum) / N;
        *Y_ptr_underlying = std::nearbyint(y_float);
      }
    });
  });
}

void qstd_inner_dim_kernel(
    const Tensor& self,
    OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction_opt,
    bool keepdim,
    Tensor& result) {
  ScalarType dtype = self.scalar_type();
  auto in_dims = self.sizes().vec();
  auto out_dims = in_dims;
  size_t num_dims_to_squeeze = dim.has_value() && !dim.value().empty() ?
                               dim.value().size() :
                               self.dim();
  int64_t M = 1; // Num of groups
  int64_t N = 1; // Num of elements to take std of in each group
  for (size_t i = 0; i < in_dims.size() - num_dims_to_squeeze; ++i) {
    M *= in_dims[i];
  }
  for (size_t i = 0; i < num_dims_to_squeeze; ++i) {
    auto idx = out_dims.size() - 1 - i;
    N *= out_dims[idx];
    out_dims[idx] = 1;
  }
  if (!keepdim) {
    out_dims.erase(out_dims.end() - num_dims_to_squeeze, out_dims.end());
  }
  const auto correction = correction_opt.value_or(1).toDouble();
  double den = std::max(N - correction, 0.0); // Denominator when computing mean and deviation
  auto x_scale = self.q_scale();
  auto x_zp = self.q_zero_point();
  result = at::_empty_affine_quantized(
      out_dims,
      at::device(kCPU).dtype(dtype).memory_format(self.suggest_memory_format()),
      x_scale,
      x_zp,
      std::nullopt);

  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "quantized_std_kernel_impl_cpu", [&]() {
    scalar_t* X_data = self.data_ptr<scalar_t>();
    scalar_t* Y_data = result.data_ptr<scalar_t>();

    at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
      for (const auto i : c10::irange(start, end)) {
        scalar_t* X_ptr = X_data + i * N;
        scalar_t* Y_ptr = Y_data + i;
        scalar_t::underlying* X_ptr_underlying = reinterpret_cast<scalar_t::underlying*>(X_ptr);
        scalar_t::underlying* Y_ptr_underlying = reinterpret_cast<scalar_t::underlying*>(Y_ptr);
        auto x_sum_shifted = hsum(X_ptr_underlying, N);
        auto x_sum_sq_shifted = hsum_sq(X_ptr_underlying, N);
        // Use double for intermediate variables to avoid accuracy issue
        // Mean with zero point
        double x_mean_shifted_div_scale_x = static_cast<double>(x_sum_shifted) / N;
        double x_mean_unbiased_shifted_div_scale_x = static_cast<double>(x_sum_shifted) / den;
        // variance / x_scale^2
        double x_var_div_scale_x_sq =
            std::max(static_cast<double>(x_sum_sq_shifted) / den -
                2 * x_mean_shifted_div_scale_x * x_mean_unbiased_shifted_div_scale_x +
                x_mean_shifted_div_scale_x * x_mean_shifted_div_scale_x * N / den, (double)0.0);
        double y_float = std::sqrt(x_var_div_scale_x_sq) * x_scale;
        *Y_ptr_underlying = at::native::quantize_val<scalar_t>(
                            x_scale, x_zp, y_float)
                            .val_;
      }
    });
  });
}

// For group norm of channels_last input
void quantized_groupnorm_nhwc_kernel(
    const Tensor& X, // input tensor
    const Tensor& gamma, // weight (optional)
    const Tensor& beta, // bias (optional)
    bool affine_per_channel, // must be true for group/instance norm
    int num_channels, // only used if affine_per_channel is set
    int num_groups, // only used if affine_per_channel is set
    int64_t M, // number of groups = Bs * G
    int64_t N, // number of elements in each group = C * H * W / G
    double eps,
    Tensor* Y) {
  AT_DISPATCH_QINT_TYPES(X.scalar_type(), "quantized_norm_nhwc_kernel_impl_cpu", [&]() {
    using qVec = vec::Vectorized<scalar_t>;
    using fVec = vec::Vectorized<float>;

    int64_t G = num_groups;
    int64_t Bs = M / G;
    int64_t C = num_channels;

    TORCH_INTERNAL_ASSERT(X.numel() == M * N, "Unexpected num elements in X");
    TORCH_INTERNAL_ASSERT(
        !gamma.defined() ||
        (!affine_per_channel && gamma.numel() == N) ||
        (affine_per_channel && gamma.numel() == C),
        "Unexpected size of gamma");
    TORCH_INTERNAL_ASSERT(
        !beta.defined() ||
        (!affine_per_channel && beta.numel() == N) ||
        (affine_per_channel && beta.numel() == C),
        "Unexpected size of beta");

    scalar_t* X_data = X.data_ptr<scalar_t>();
    const float* gamma_data = gamma.defined() ? gamma.const_data_ptr<float>() : nullptr;
    const float* beta_data = beta.defined() ? beta.const_data_ptr<float>() : nullptr;
    scalar_t* Y_data = Y->data_ptr<scalar_t>();
    const bool gamma_null = gamma_data == nullptr;
    const bool beta_null = beta_data == nullptr;
    int64_t x_zp = X.q_zero_point();
    float x_scale = X.q_scale();
    fVec x_zp_vec(x_zp);
    fVec one_vec(1.0f);
    fVec zero_vec(0.0f);
    float x_fake_scale = 1.0f;
    fVec x_fake_scale_vec(x_fake_scale);
    fVec x_fake_scale_zp_neg_premul_vec = x_fake_scale_vec * x_zp_vec.neg();
    int64_t y_zp = Y->q_zero_point();
    float y_scale = Y->q_scale();
    float y_inv_scale = 1.0f / y_scale;

    constexpr int kFloatVLen = fVec::size();
    int64_t kIntVLen = kFloatVLen * qVec::float_num_vecs();
    int64_t channels_per_group = C / G;
    int64_t HxW = N / channels_per_group;
    int64_t kNumIntVecInHxW = channels_per_group / kIntVLen;
    int64_t kNonVecRemInHxW = channels_per_group % kIntVLen;
    int64_t kNumIntVecOnChannel = C / kIntVLen;
    int64_t kNonVecRemOnChannel = C % kIntVLen;

    // Buffer for x and x^2
    Tensor buffer = at::empty({M, 2 * channels_per_group}, X.options().dtype(at::kFloat));
    float* buffer_data = buffer.mutable_data_ptr<float>();

    // We can parallel in the following 2 impls:
    //
    // impl-1: parallel on N * G. Only need one omp session but memory access
    //   per thread is non-contiguous.
    //
    // impl-2: parallel on N * HxW. Memory access per thread is contiguous,
    //   but requires help of extra temp buffer of size {T, N, 2C}.
    //
    // Generally impl-2 has better performance when HxW is large enough
    // The threshold is found by tests.
    constexpr int64_t feature_map_threshold = 512;
    if (HxW < feature_map_threshold) {
      // Impl-1: Parallel for each group
      //
      // Parallel for each group, M = Bs * G
      at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
        int64_t n{0} /* batch index */, g{0} /* group index in each batch */;
        data_index_init(begin, n, N, g, G);
        for (const auto grpIdx : c10::irange(begin, end)) { // For each group

          // Step 1: calculate mean and variance.
          int64_t l_sum_shifted = 0;
          int64_t l_sum_sq_shifted = 0;
          for (const auto hw : c10::irange(HxW)) {
            scalar_t* X_ptr = X_data + n * N * G + g * channels_per_group + hw * C;
            scalar_t::underlying* X_ptr_underlying = reinterpret_cast<scalar_t::underlying*>(X_ptr);
            l_sum_shifted += hsum(X_ptr_underlying, channels_per_group);
            l_sum_sq_shifted += hsum_sq(X_ptr_underlying, channels_per_group);
          }

          // mean(dqX) / scale_x + x_zp
          float l_mean_shifted_div_scale_x = static_cast<float>(l_sum_shifted) / N;
          // mean(dqX) / scale_x
          float layer_mean_div_scale_x = l_mean_shifted_div_scale_x - x_zp;
          // var(dqX) / scale_x^2
          float layer_var_div_scale_x_sq =
            std::max(static_cast<float>(l_sum_sq_shifted) / N -
                l_mean_shifted_div_scale_x * l_mean_shifted_div_scale_x, 0.0f);
          // scale_x / sqrt(var(dqX) + eps)
          float scale_x_div_layer_std = x_scale /
            std::sqrt(layer_var_div_scale_x_sq * x_scale * x_scale + eps);

          // Step 2: calculate scale and bias
          float* scale_ptr = buffer_data + grpIdx * 2 * channels_per_group;
          float* bias_ptr = scale_ptr + channels_per_group;
          for (const auto d : c10::irange(channels_per_group)) {
            const int64_t chIdx = g * channels_per_group + d;
            scale_ptr[d] = scale_x_div_layer_std * (gamma_null ? 1.0f : gamma_data[chIdx]);
            bias_ptr[d] = -scale_ptr[d] * layer_mean_div_scale_x + (beta_null ? 0.0f : beta_data[chIdx]);
          }

          // Step 3: applying scale and bias
          for (const auto hwIdx : c10::irange(HxW)) {
            const scalar_t* X_ptr = X_data + n * N * G + g * channels_per_group + hwIdx * C;
            scalar_t* Y_ptr = Y_data + n * N * G + g * channels_per_group + hwIdx * C;
            // vectorized
            for (const auto vecIdx : c10::irange(kNumIntVecInHxW)) {
              int64_t vecStartIdx = vecIdx * kIntVLen;
              auto qXVec = qVec::loadu(X_ptr + vecStartIdx);
              auto dqXVec = qXVec.dequantize(x_fake_scale_vec, x_zp_vec,
                    x_fake_scale_zp_neg_premul_vec);
              for (size_t fvecIdx = 0; fvecIdx < dqXVec.size(); ++fvecIdx) {
                auto scaleVec = fVec::loadu(scale_ptr + vecStartIdx + fvecIdx * kFloatVLen);
                auto biasVec = fVec::loadu(bias_ptr + vecStartIdx + fvecIdx * kFloatVLen);
                dqXVec[fvecIdx] = dqXVec[fvecIdx] * scaleVec + biasVec;
              }
              qVec::quantize(dqXVec, y_scale, y_zp, y_inv_scale)
                  .store(Y_ptr + vecStartIdx);
            }
            // Remaining scalar
            for (int64_t remIdx = kNumIntVecInHxW * kIntVLen;
                 remIdx < kNonVecRemInHxW + kNumIntVecInHxW * kIntVLen;
                 ++remIdx) {
              auto qXVal = X_ptr[remIdx];
              float dqXVal = at::native::dequantize_val(x_fake_scale, x_zp, qXVal);
              float dqY = dqXVal * scale_ptr[remIdx] + bias_ptr[remIdx];
              Y_ptr[remIdx] = at::native::quantize_val<scalar_t>(y_scale, y_zp, dqY);
            }
          } // loop over HxW

          data_index_step(n, N, g, G);
        } // for each group
      }); // parallel_for
    } else { // HxW > feature_map_threshold
      // impl-2: parallel on Bs * HxW.
      //
      // Buffer for x and x^2
      // To avoid thread conflict, we use a temp buffer of {T, Bs, 2*C}
      int num_threads = at::get_num_threads();
      Tensor buffer = at::empty({num_threads, Bs, 2 * C}, X.options().dtype(at::kFloat)).zero_();
      float* buffer_data = buffer.mutable_data_ptr<float>();
      Tensor mean = at::empty(M, X.options().dtype(at::kFloat));
      float* mean_data = mean.mutable_data_ptr<float>();
      Tensor rstd = at::empty(M, X.options().dtype(at::kFloat));
      float* rstd_data = rstd.mutable_data_ptr<float>();

      // Step 1: Accumulate on C dimension
      at::parallel_for(0, Bs * HxW, 1, [&](int64_t begin, int64_t end) {
        int tid = at::get_thread_num();
        float* buffer_ptr = buffer_data + tid * Bs * 2 * C;

        int64_t n{0} /* batch index */, m{0} /* HxW index */;
        data_index_init(begin, n, Bs, m, HxW);
        for (const auto nhwIdx : c10::irange(begin, end)) {
          float* mean_ptr = buffer_ptr + n * 2 * C;
          float* rstd_ptr = mean_ptr + C;
          scalar_t* X_ptr = X_data + nhwIdx * C;
          scalar_t::underlying* X_ptr_underlying = reinterpret_cast<scalar_t::underlying*>(X_ptr);
          for (int chIdx = 0; chIdx < C; ++chIdx) {
            auto x = X_ptr_underlying[chIdx];
            mean_ptr[chIdx] += x;
            rstd_ptr[chIdx] += x * x;
          }
          data_index_step(n, Bs, m, HxW);
        }
      });

      // Step 2: Calculate mean and rstd
      for (const auto n : c10::irange(Bs)) {
        for (const auto g : c10::irange(G)) {
          float mean_val{0}, rstd_val{0};
          for (const auto t : c10::irange(num_threads)) {
            float* buffer_ptr = buffer_data + t * Bs * 2 * C + n * 2 * C;
            for (const auto d : c10::irange(channels_per_group)) {
              mean_val += buffer_ptr[g * channels_per_group + d];
              rstd_val += buffer_ptr[g * channels_per_group + d + C];
            } // for d
          } // for t

          // mean / scale_x + x_zp
          float l_mean_shifted_div_scale_x = mean_val / N;
          // mean / scale_x
          float layer_mean_div_scale_x = l_mean_shifted_div_scale_x - x_zp;
          // var / scale_x^2
          float layer_var_div_scale_x_sq =
              std::max(rstd_val / N -
              l_mean_shifted_div_scale_x * l_mean_shifted_div_scale_x, 0.0f);
          // scale_x / sqrt(var + eps)
          float scale_x_div_layer_std = x_scale /
              std::sqrt(layer_var_div_scale_x_sq * x_scale * x_scale + eps);
          mean_data[n * G + g] = layer_mean_div_scale_x;
          rstd_data[n * G + g] = scale_x_div_layer_std;

        } // for g
      } // for n

      // Step 3: Calculate scale and bias
      //
      // We could fuse step 3 and 4 into a single session but this way is better:
      //   a. D might be too small for vectorization;
      //   b. Avoid duplicate calculation of scale/bias, each HxW plain share the same scale/bias
      //
      for (const auto n : c10::irange(Bs)) {
        for (const auto g : c10::irange(G)) {
          float* scale_ptr = buffer_data + n * 2 * C;
          float* bias_ptr = scale_ptr + C;
          float mean_val = mean_data[n * G + g];
          float rstd_val = rstd_data[n * G + g];
          for (const auto d : c10::irange(channels_per_group)) {
            const int64_t chIdx = g * channels_per_group + d;
            scale_ptr[chIdx] = rstd_val * (gamma_null ? 1.0f : gamma_data[chIdx]);
            bias_ptr[chIdx] = -scale_ptr[chIdx] * mean_val + (beta_null ? 0.0f : beta_data[chIdx]);
          } // for d
        } // for g
      } // for n

      // step-4: apply scale and bias
      //
      // Parallel on all the outer dimensions of Bs and HxW
      // and vectorize on C.
      //
      at::parallel_for(0, Bs * HxW, 1, [&](int64_t begin, int64_t end) {
        int64_t n{0}, m{0};
        data_index_init(begin, n, Bs, m, HxW);
        for (const auto nhwIdx : c10::irange(begin, end)) {
          const scalar_t* X_ptr = X_data + nhwIdx * C;
          scalar_t* Y_ptr = Y_data + nhwIdx * C;
          float* scale_ptr = buffer_data + n * 2 * C;
          float* bias_ptr = scale_ptr + C;
          // Vectorized
          for (const auto vecIdx : c10::irange(kNumIntVecOnChannel)) {
            int64_t vecStartIdx = vecIdx * kIntVLen;
            auto qXVec = qVec::loadu(X_ptr + vecStartIdx);
            auto dqXVec = qXVec.dequantize(x_fake_scale_vec, x_zp_vec,
                  x_fake_scale_zp_neg_premul_vec);
            for (size_t fvecIdx = 0; fvecIdx < dqXVec.size(); ++fvecIdx) {
              auto scaleVec = fVec::loadu(scale_ptr + vecStartIdx + fvecIdx * kFloatVLen);
              auto biasVec = fVec::loadu(bias_ptr + vecStartIdx + fvecIdx * kFloatVLen);
              dqXVec[fvecIdx] = dqXVec[fvecIdx] * scaleVec + biasVec;
            }
            qVec::quantize(dqXVec, y_scale, y_zp, y_inv_scale)
                .store(Y_ptr + vecStartIdx);
          }
          // Remaining scalar
          for (int64_t remIdx = kNumIntVecOnChannel * kIntVLen;
               remIdx < kNonVecRemOnChannel + kNumIntVecOnChannel * kIntVLen;
               ++remIdx) {
            auto qXVal = X_ptr[remIdx];
            float dqXVal = at::native::dequantize_val(x_fake_scale, x_zp, qXVal);
            float dqY = dqXVal * scale_ptr[remIdx] + bias_ptr[remIdx];
            Y_ptr[remIdx] = at::native::quantize_val<scalar_t>(y_scale, y_zp, dqY);
          }

          data_index_step(n, Bs, m, HxW);
        } // for idx on nhw
      }); // parallel_for on nhw

    } // if HxW > feature_map_threshold

  }); // AT_DISPATCH_QINT_TYPES
}

#ifdef USE_FBGEMM
void quantize_tensor_per_tensor_affine_cpu(
    const Tensor& rtensor,
    Tensor& qtensor,
    double scale,
    int64_t zero_point) {
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "quantize_tensor_per_tensor_affine_cpu", [&]() {
        check_tensor_memory_format(rtensor, qtensor);
        const float* rd = rtensor.const_data_ptr<float>();
        auto qd = reinterpret_cast<underlying_t*>(qtensor.data_ptr<scalar_t>());
        fbgemm::TensorQuantizationParams qparams{};
        qparams.scale = scale;
        qparams.zero_point = zero_point;
        qparams.precision = CHAR_BIT * sizeof(underlying_t);
        int num_tasks = at::get_num_threads();
        at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
          for (const auto task_id : c10::irange(begin, end)) {
            fbgemm::Quantize<underlying_t, false /*LEGACY*/>(
                rd, /*src=*/
                qd, /*dst=*/
                rtensor.numel(), /*len*/
                qparams, /*qparams=*/
                task_id, /*thread_id*/
                num_tasks /*num_threads*/);
          }
        });
      });
}

void dequantize_tensor_per_tensor_affine_cpu(
    const Tensor& qtensor,
    Tensor& rtensor,
    double scale,
    int64_t zero_point) {
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "dequantize_tensor_per_tensor_affine_cpu", [&]() {
        check_tensor_memory_format(qtensor, rtensor);
        const auto* qd =
            reinterpret_cast<const underlying_t*>(qtensor.data_ptr<scalar_t>());
        fbgemm::TensorQuantizationParams qparams{};
        qparams.scale = scale;
        qparams.zero_point = zero_point;
        qparams.precision = CHAR_BIT * sizeof(underlying_t);
        float* rd = rtensor.data_ptr<float>();
        int num_tasks = at::get_num_threads();
        at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
          for (const auto task_id : c10::irange(begin, end)) {
            fbgemm::Dequantize<underlying_t>(
                qd, /*src=*/
                rd, /*dst=*/
                qtensor.numel(), /*len=*/
                qparams, /*qparams=*/
                task_id, /*thread_id*/
                num_tasks /*num_threads*/);
          }
        });
      });
}
#else // USE_FBGEMM

#if defined(__ARM_NEON__) || defined(__aarch64__)

constexpr static int PARALLEL_THRESHOLD = 1 << 20;

// Generic template defaults to naive quantize implementation
template <typename T>
void quantize_tensor_arm(
    const float* __restrict__ in,
    T* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  for (const auto i : c10::irange(N)) {
    out[i] = at::native::quantize_val<T>(scale, zero_point, in[i]);
  }
}

namespace quantize_tensor_arm_intrinsics {
template <typename Tx8>
C10_ALWAYS_INLINE Tx8 vqmov(int16x8_t vraw);

template <>
C10_ALWAYS_INLINE uint8x8_t vqmov<uint8x8_t>(int16x8_t vraw) {
  return vqmovun_s16(vraw);
}

template <>
C10_ALWAYS_INLINE int8x8_t vqmov<int8x8_t>(int16x8_t vraw) {
  return vqmovn_s16(vraw);
}

template <typename T, typename Tx8>
C10_ALWAYS_INLINE void vst1(T* out, Tx8 vout);

template <>
C10_ALWAYS_INLINE void vst1<uint8_t, uint8x8_t>(uint8_t* out, uint8x8_t vout) {
  vst1_u8(out, vout);
}

template <>
C10_ALWAYS_INLINE void vst1<int8_t, int8x8_t>(int8_t* out, int8x8_t vout) {
  vst1_s8(out, vout);
}
} // namespace quantize_tensor_arm_intrinsics

// Specialized implementation from caffe2::Int8Quantize.
// There may be slight accuracy difference between this and implementation of
// quantize_val
// TODO Update quantize_tensor_arm implementation to follow quantize_val,
// i.e. f = Round(value/scale + zero_point)
// TODO Make quantize_tensor_arm work for int32 datatype too.
template <typename scalar_t, typename underlying_t, typename underlying_x8_t>
void quantize_tensor_arm_q8(
    const float* __restrict__ in,
    scalar_t* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  const float inv_scale = 1.0f / scale;
  uint32_t i = 0;
  underlying_t* out_underlying = reinterpret_cast<underlying_t*>(out);
  const float32x4_t vinv_scale = vdupq_n_f32(inv_scale);
#if defined(__ARM_NEON__)
  // magic float and magic int to take care of rounding
  // int magic_round(float f): interpret_int32(f + 12582912.0f) - 0x4B400000
  // Some detail:
  // 12582912.0f is 2**23 + 2**22. The trick is based on the fact that when you
  // add a small number to a large number, the result rounds to the precision of
  // the least significant bit of the large number. For IEEE-754
  // single-precision number mantissa has 23 bits, and adding 2**23 would cause
  // rounding to the nearest even integer. The we cast to int and subtract the
  // same number (0x4B400000 is the integer representation of 12582912.0f) to
  // get only the mantissa. This works if -2**22 < x < 2**22, but preserves the
  // sign for negative numbers.
  const int32x4_t voffset = vdupq_n_s32(zero_point - 0x4B400000);
  const float32x4_t vmagic_float = vdupq_n_f32(12582912.0f);
  for (i = 0; i + 8 <= N; i += 8) {
    const float32x4_t vin0123 = vld1q_f32(in);
    in += 4;
    const float32x4_t vin4567 = vld1q_f32(in);
    in += 4;
    const int32x4_t vraw0123 = vaddq_s32(
        voffset,
        vreinterpretq_s32_f32(
            vaddq_f32(vmagic_float, vmulq_f32(vin0123, vinv_scale))));
    const int32x4_t vraw4567 = vaddq_s32(
        voffset,
        vreinterpretq_s32_f32(
            vaddq_f32(vmagic_float, vmulq_f32(vin4567, vinv_scale))));
    const int16x8_t vraw01234567 =
        vcombine_s16(vqmovn_s32(vraw0123), vqmovn_s32(vraw4567));
    const underlying_x8_t vout01234567 =
        quantize_tensor_arm_intrinsics::vqmov<underlying_x8_t>(vraw01234567);
    quantize_tensor_arm_intrinsics::vst1<underlying_t, underlying_x8_t>(
        out_underlying, vout01234567);
    out_underlying += 8;
  }
  for (; i < N; ++i) {
    (*out_underlying++) =
        at::native::quantize_val_arm<underlying_t>(scale, zero_point, (*in++));
  }
#else
  const int16x8_t vzero_point = vdupq_n_s16((int16_t)(uint16_t)zero_point);
  for (i = 0; i + 8 <= N; i += 8) {
    const float32x4_t vin0123 = vld1q_f32(in);
    in += 4;
    const float32x4_t vin4567 = vld1q_f32(in);
    in += 4;
    const int32x4_t v0123_rounded = vcvtnq_s32_f32(vmulq_f32(vin0123, vinv_scale));
    const int32x4_t v4567_rounded = vcvtnq_s32_f32(vmulq_f32(vin4567, vinv_scale));
    const int16x8_t v01234567_packed = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(v0123_rounded), v4567_rounded), vzero_point);
    const underlying_x8_t vout01234567 =
        quantize_tensor_arm_intrinsics::vqmov<underlying_x8_t>(
            v01234567_packed);
    quantize_tensor_arm_intrinsics::vst1<underlying_t, underlying_x8_t>(
        out_underlying, vout01234567);
    out_underlying += 8;
  }
  for (; i < N; ++i) {
    (*out_underlying++) =
        at::native::quantize_val_arm<underlying_t>(scale, zero_point, (*in++));
  }
#endif
}

template <>
void quantize_tensor_arm<c10::quint8>(
    const float* __restrict__ in,
    c10::quint8* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  quantize_tensor_arm_q8<c10::quint8, uint8_t, uint8x8_t>(
      in, out, N, scale, zero_point);
}

template <>
void quantize_tensor_arm<c10::qint8>(
    const float* __restrict__ in,
    c10::qint8* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  quantize_tensor_arm_q8<c10::qint8, int8_t, int8x8_t>(
      in, out, N, scale, zero_point);
}

#if defined(__aarch64__)
#define VMOVL_HIGH_U8(x) vmovl_high_u8(x)
#define VMOVL_HIGH_S8(x) vmovl_high_s8(x)
#define VMOVL_HIGH_U16(x) vmovl_high_u16(x)
#define VMOVL_HIGH_S16(x) vmovl_high_s16(x)
#else // vmovl_high intrinsic not supported
#define VMOVL_HIGH_U8(x) vmovl_u8(vget_high_u8(x))
#define VMOVL_HIGH_S8(x) vmovl_s8(vget_high_s8(x))
#define VMOVL_HIGH_U16(x) vmovl_u16(vget_high_u16(x))
#define VMOVL_HIGH_S16(x) vmovl_s16(vget_high_s16(x))
#endif

// Generic template defaults to naive dequantize implementation
template <typename T>
void dequantize_tensor_arm(
    const T* __restrict__ in,
    float* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  for (int i = 0; i < N; ++i) {
    out[i] = dequantize_val<T>(scale, zero_point, in[i]);
  }
}

template <>
void dequantize_tensor_arm<c10::qint8>(
    const c10::qint8* __restrict__ in,
    float* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  const int8_t* in_underlying = reinterpret_cast<const int8_t*>(in);

  const float32x4_t scale_fp32x4 = vdupq_n_f32(scale);
  // Zero point is restricted to be in bounds of a signed 8 bit integer
  const int8x8_t zero_point_s8x8 = vget_low_s8(vdupq_n_s8(static_cast<int8_t>(zero_point)));

  int i;
  for (i = 0; i + 16 <= N; i += 16) {
    const int8x16_t vin_s8 = vld1q_s8(in_underlying);

    // Extract upper or lower values to int16x8 and subtract zero point
    // Each input element and the zero point are restricted to be in bounds of
    // a signed 8 bit integer, so the difference will fit in a signed 16 bit
    // integer
    const int16x8_t minus_zp_low_s16 = vsubl_s8(vget_low_s8(vin_s8), zero_point_s8x8); // 0 ... 7
    const int16x8_t minus_zp_high_s16 = vsubl_s8(vget_high_s8(vin_s8), zero_point_s8x8); // 8 ... 15

    const int32x4_t minus_zp_low_low = vmovl_s16(vget_low_s16(minus_zp_low_s16)); // 0 ... 3
    const int32x4_t minus_zp_low_high = VMOVL_HIGH_S16(minus_zp_low_s16); // 4 ... 7
    const int32x4_t minus_zp_high_low = vmovl_s16(vget_low_s16(minus_zp_high_s16)); // 8 ... 11
    const int32x4_t minus_zp_high_high = VMOVL_HIGH_S16(minus_zp_high_s16); // 12 ... 15

    // Store            * scale   int32->fp32
    vst1q_f32(out,      vmulq_f32(vcvtq_f32_s32(minus_zp_low_low), scale_fp32x4));
    vst1q_f32(out + 4,  vmulq_f32(vcvtq_f32_s32(minus_zp_low_high), scale_fp32x4));
    vst1q_f32(out + 8,  vmulq_f32(vcvtq_f32_s32(minus_zp_high_low), scale_fp32x4));
    vst1q_f32(out + 12, vmulq_f32(vcvtq_f32_s32(minus_zp_high_high), scale_fp32x4));

    out += 16;
    in += 16;
    in_underlying += 16;
  }

  for (; i < N; ++i) { // use default dequantize for remaining vals
    (*out++) = dequantize_val<c10::qint8>(scale, zero_point, (*in++));
  }
}

template <>
void dequantize_tensor_arm<c10::quint8>(
    const c10::quint8* __restrict__ in,
    float* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  const uint8_t* in_underlying = reinterpret_cast<const uint8_t*>(in);

  const float32x4_t scale_fp32x4 = vdupq_n_f32(scale);
  // Zero point is restricted to be in bounds of an unsigned 8 bit integer
  const uint8x8_t zero_point_u8x8 = vget_low_u8(vdupq_n_u8(static_cast<uint8_t>(zero_point)));

  int i;
  for (i = 0; i + 16 <= N; i += 16) {
    const uint8x16_t vin_u8 = vld1q_u8(in_underlying);

    // Extract upper or lower values to uint16x8 and subtract zero point
    // Each input element and the zero point are restricted to be in bounds of
    // an unsigned 8 bit integer, so the difference will fit in a signed 16 bit
    // integer
    const int16x8_t minus_zp_low_s16 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vin_u8), zero_point_u8x8)); // 0 ... 7
    const int16x8_t minus_zp_high_s16 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vin_u8), zero_point_u8x8)); // 8 ... 15

    const int32x4_t minus_zp_low_low = vmovl_s16(vget_low_s16(minus_zp_low_s16)); // 0 ... 3
    const int32x4_t minus_zp_low_high = VMOVL_HIGH_S16(minus_zp_low_s16); // 4 ... 7
    const int32x4_t minus_zp_high_low = vmovl_s16(vget_low_s16(minus_zp_high_s16)); // 8 ... 11
    const int32x4_t minus_zp_high_high = VMOVL_HIGH_S16(minus_zp_high_s16); // 12 ... 15

    // Store            * scale   int32->fp32
    vst1q_f32(out,      vmulq_f32(vcvtq_f32_s32(minus_zp_low_low), scale_fp32x4));
    vst1q_f32(out + 4,  vmulq_f32(vcvtq_f32_s32(minus_zp_low_high), scale_fp32x4));
    vst1q_f32(out + 8,  vmulq_f32(vcvtq_f32_s32(minus_zp_high_low), scale_fp32x4));
    vst1q_f32(out + 12, vmulq_f32(vcvtq_f32_s32(minus_zp_high_high), scale_fp32x4));

    out += 16;
    in += 16;
    in_underlying += 16;
  }

  for (; i < N; ++i) { // use default dequantize for remaining vals
    (*out++) = dequantize_val<c10::quint8>(scale, zero_point, (*in++));
  }
}

#endif // defined(__ARM_NEON__) || defined(__aarch64__)

void quantize_tensor_per_tensor_affine_cpu(
    const Tensor& rtensor,
    Tensor& qtensor,
    double scale,
    int64_t zero_point) {
  check_tensor_memory_format(rtensor, qtensor);
  const float* rdata = rtensor.const_data_ptr<float>();
  int numel = rtensor.numel();
#if defined(__ARM_NEON__) || defined(__aarch64__)
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "quantize_tensor_per_tensor_affine_cpu", [&]() {
        scalar_t* qdata = qtensor.data_ptr<scalar_t>();
        auto quantize_range = [&](int64_t begin, int64_t end) {
          quantize_tensor_arm<scalar_t>(
            rdata + begin, qdata + begin, end - begin, scale, zero_point);
        };
        if (numel >= PARALLEL_THRESHOLD) {
          at::parallel_for(0, numel, 1, quantize_range);
        } else {
          quantize_range(0, numel);
        }
      });
#else
  // Fallback path
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "quantize_tensor_per_tensor_affine_cpu", [&]() {
        scalar_t* qdata = qtensor.data_ptr<scalar_t>();
        for (const auto i : c10::irange(numel)) {
          qdata[i] = quantize_val<scalar_t>(scale, zero_point, rdata[i]);
        }
      });
#endif // defined(__ARM_NEON__) || defined(__aarch64__)
}

void dequantize_tensor_per_tensor_affine_cpu(
    const Tensor& qtensor,
    Tensor& rtensor,
    double scale,
    int64_t zero_point) {
  check_tensor_memory_format(qtensor, rtensor);
  float* rdata = rtensor.data_ptr<float>();
  int numel = qtensor.numel();
#if defined(__ARM_NEON__) || defined(__aarch64__)
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "dequantize_tensor_per_tensor_affine_cpu", [&]() {
        const scalar_t* qdata = qtensor.const_data_ptr<scalar_t>();
        auto dequantize_range = [&](int64_t begin, int64_t end) {
          dequantize_tensor_arm<scalar_t>(
            qdata + begin, rdata + begin, end - begin, scale, zero_point);
        };
        if (numel >= PARALLEL_THRESHOLD) {
          at::parallel_for(0, numel, 1, dequantize_range);
        } else {
          dequantize_range(0, numel);
        }
      });
#else
  // Fallback path
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "dequantize_tensor_per_tensor_affine_cpu", [&]() {
        const scalar_t* qdata = qtensor.const_data_ptr<scalar_t>();
        for (const auto i : c10::irange(numel)) {
          rdata[i] = dequantize_val<scalar_t>(scale, zero_point, qdata[i]);
        }
      });
#endif // defined(__ARM_NEON__) || defined(__aarch64__)
}
#endif // USE_FBGEMM

// TODO: add fbgemm for per channel
// Generic template defaults to naive quantize implementation
template <typename T>
void quantize_tensor_per_channel_impl(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  // TODO: channels last kernel can be made faster.
  // For contiguous tensors, e.g. NCHW, arbitrary axis can be used.
  // For channels_last/3d however axis == 0 or 1.
  // Since current implementation on channels_last format does not
  // cover per channel quant with arbitrary axis value, it is better
  // to check and fail.
  int64_t batches = size_to_dim_(axis, rtensor.sizes());
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  int64_t elements_per_channel = size_from_dim_(axis + 1, rtensor.sizes());
  int64_t channels = rtensor.size(axis);
  auto scales_data = scales.data_ptr<double>();
  auto zero_points_data = zero_points.data_ptr<int64_t>();
  const float* in = rtensor.const_data_ptr<float>();
  auto out = qtensor.data_ptr<T>();
  if (axis == 1 &&
      (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
       rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
    // This code handles per channel quant when axis = 1 and
    // channels_last contig.
    // If axis = 0 and channels_last contig, implementation for channels
    // first (NCHW) works.
    for (const auto b : c10::irange(batches)) {
      for (const auto e : c10::irange(elements_per_channel)) {
        for (const auto c : c10::irange(channels)) {
          auto i = b * channels * elements_per_channel + e * channels + c;
          out[i] = at::native::quantize_val<T>(
              scales_data[c], zero_points_data[c], in[i]);
        }
      }
    }
  } else {
    for (const auto b : c10::irange(batches)) {
      for (const auto c : c10::irange(channels)) {
        for (const auto e : c10::irange(elements_per_channel)) {
          auto i = b * channels * elements_per_channel +
              c * elements_per_channel + e;
          out[i] = at::native::quantize_val<T>(
              scales_data[c], zero_points_data[c], in[i]);
        }
      }
    }
  }
}

#if defined(__ARM_NEON__) || defined(__aarch64__)
// Specialized implementation from caffe2::Int8Quantize.
// There may be slight accuracy difference between this and implementation of
// quantize_val
// TODO Update quantize_tensor_per_channel_impl implementation to follow
// quantize_val, i.e. f = Round(value/scale + zero_point)
// TODO Make quantize_tensor_per_channel_impl work for other datatypes too
// (int8, int32).
template <>
void quantize_tensor_per_channel_impl<c10::quint8>(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  int64_t batches = size_to_dim_(axis, rtensor.sizes());
  int64_t elements_per_channel = size_from_dim_(axis + 1, rtensor.sizes());
  int64_t channels = rtensor.size(axis);
  auto scales_data = scales.data_ptr<double>();
  auto zero_points_data = zero_points.data_ptr<int64_t>();
  const float* in = rtensor.const_data_ptr<float>();
  auto out = (uint8_t*)qtensor.data_ptr<c10::quint8>();
#if defined(__ARM_NEON__)
  // magic float and magic int to take care of rounding
  // int magic_round(float f): interpret_int32(f + 12582912.0f) - 0x4B400000
  // Some detail:
  // 12582912.0f is 2**23 + 2**22. The trick is based on the fact that when you
  // add a small number to a large number, the result rounds to the precision of
  // the least significant bit of the large number. For IEEE-754
  // single-precision number mantissa has 23 bits, and adding 2**23 would cause
  // rounding to the nearest even integer. The we cast to int and subtract the
  // same number (0x4B400000 is the integer representation of 12582912.0f) to
  // get only the mantissa. This works if -2**22 < x < 2**22, but preserves the
  // sign for negative numbers.
  const float32x4_t vmagic_float = vdupq_n_f32(12582912.0f);
  // Copy reciprocal of scales (double) into float array
  // Copy zero_points with magic int (int64_t) into int32_t array
  std::vector<float> inv_scales(channels);
  std::vector<int32_t> zero_points_int32t(channels);
  for (const auto i : c10::irange(channels)) {
    inv_scales[i] = 1.0f / (float)scales_data[i];
    zero_points_int32t[i] = (int32_t)(uint32_t)zero_points_data[i] - 0x4B400000;
  }
  if (axis == 1 &&
      (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
       rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
    // This code handles per channel quant when axis = 1 and
    // channels_last contig.
    // If axis = 0 and channels_last contig, implementation for channels
    // first (NCHW) works.
    for ([[maybe_unused]] const auto b : c10::irange(batches)) {
      for ([[maybe_unused]] const auto e : c10::irange(elements_per_channel)) {
        uint32_t c = 0;
        while (c + 8 < channels) {
          const int32x4_t voffset0123 = vld1q_s32(&zero_points_int32t[c]);
          const float32x4_t vinv_scale0123 = vld1q_f32(&inv_scales[c]);
          c += 4;
          const int32x4_t voffset4567 = vld1q_s32(&zero_points_int32t[c]);
          const float32x4_t vinv_scale4567 = vld1q_f32(&inv_scales[c]);
          c += 4;
          const float32x4_t vin0123 = vld1q_f32(in);
          in += 4;
          const float32x4_t vin4567 = vld1q_f32(in);
          in += 4;
          const int32x4_t vraw0123 = vaddq_s32(
              voffset0123,
              vreinterpretq_s32_f32(
                  vaddq_f32(vmagic_float, vmulq_f32(vin0123, vinv_scale0123))));
          const int32x4_t vraw4567 = vaddq_s32(
              voffset4567,
              vreinterpretq_s32_f32(
                  vaddq_f32(vmagic_float, vmulq_f32(vin4567, vinv_scale4567))));
          const int16x8_t vraw01234567 =
              vcombine_s16(vqmovn_s32(vraw0123), vqmovn_s32(vraw4567));
          const uint8x8_t vout01234567 = vqmovun_s16(vraw01234567);
          vst1_u8(out, vout01234567);
          out += 8;
        }
        for (; c < channels; ++c) {
          (*out++) = at::native::quantize_val_arm<uint8_t>(
              scales_data[c], zero_points_data[c], (*in++));
        }
      }
    }
  } else {
    for ([[maybe_unused]] const auto b : c10::irange(batches)) {
      for (const auto c : c10::irange(channels)) {
        uint32_t e = 0;
        const int32x4_t voffset = vdupq_n_s32(zero_points_int32t[c]);
        const float32x4_t vinv_scale = vdupq_n_f32(inv_scales[c]);
        for (; e + 8 < elements_per_channel; e += 8) {
          const float32x4_t vin0123 = vld1q_f32(in);
          in += 4;
          const float32x4_t vin4567 = vld1q_f32(in);
          in += 4;
          const int32x4_t vraw0123 = vaddq_s32(
              voffset,
              vreinterpretq_s32_f32(
                  vaddq_f32(vmagic_float, vmulq_f32(vin0123, vinv_scale))));
          const int32x4_t vraw4567 = vaddq_s32(
              voffset,
              vreinterpretq_s32_f32(
                  vaddq_f32(vmagic_float, vmulq_f32(vin4567, vinv_scale))));
          const int16x8_t vraw01234567 =
              vcombine_s16(vqmovn_s32(vraw0123), vqmovn_s32(vraw4567));
          const uint8x8_t vout01234567 = vqmovun_s16(vraw01234567);
          vst1_u8(out, vout01234567);
          out += 8;
        }
        for (; e < elements_per_channel; ++e) {
          (*out++) = at::native::quantize_val_arm<uint8_t>(
              scales_data[c], zero_points_data[c], (*in++));
        }
      }
    }
  }
#else // defined(__ARM_NEON__)
  // Copy scales (double) into float array
  // Copy zero_points (int64_t) into int16_t array
  std::vector<float> inv_scales(channels);
  std::vector<int16_t> zero_points_int16t(channels);
  for (const auto i : c10::irange(channels)) {
    inv_scales[i] = 1.0f / (float)scales_data[i];
    zero_points_int16t[i] = (int16_t)(uint16_t)zero_points_data[i];
  }
  if (axis == 1 &&
      (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
       rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
    // This code handles per channel quant when axis = 1 and
    // channels_last contig.
    // If axis = 0 and channels_last contig, implementation for channels
    // first (NCHW) works.
    for ([[maybe_unused]] const auto b : c10::irange(batches)) {
      for ([[maybe_unused]] const auto e : c10::irange(elements_per_channel)) {
        uint32_t c = 0;
        while (c + 8 < channels) {
          const int16x8_t vzero_point = vld1q_s16(&zero_points_int16t[c]);
          const float32x4_t vinv_scale0123 = vld1q_f32(&inv_scales[c]);
          c += 4;
          const float32x4_t vinv_scale4567 = vld1q_f32(&inv_scales[c]);
          c += 4;
          const float32x4_t vin0123 = vld1q_f32(in);
          in += 4;
          const float32x4_t vin4567 = vld1q_f32(in);
          in += 4;
          const int32x4_t v0123_rounded =
              vcvtnq_s32_f32(vmulq_f32(vin0123, vinv_scale0123));
          const int32x4_t v4567_rounded =
              vcvtnq_s32_f32(vmulq_f32(vin4567, vinv_scale4567));
          const int16x8_t v01234567_packed = vqaddq_s16(
              vqmovn_high_s32(vqmovn_s32(v0123_rounded), v4567_rounded),
              vzero_point);
          const uint8x8_t vout01234567 = vqmovun_s16(v01234567_packed);
          vst1_u8(out, vout01234567);
          out += 8;
        }
        for (; c < channels; ++c) {
          (*out++) = at::native::quantize_val_arm<uint8_t>(
              scales_data[c], zero_points_data[c], (*in++));
        }
      }
    }
  } else {
    for ([[maybe_unused]] const auto b : c10::irange(batches)) {
      for ([[maybe_unused]] const auto c : c10::irange(channels)) {
        uint32_t e = 0;
        const int16x8_t vzero_point = vdupq_n_s16(zero_points_int16t[c]);
        const float32x4_t vinv_scale = vdupq_n_f32(inv_scales[c]);
        for (; e + 8 < elements_per_channel; e += 8) {
          const float32x4_t vin0123 = vld1q_f32(in);
          in += 4;
          const float32x4_t vin4567 = vld1q_f32(in);
          in += 4;
          const int32x4_t v0123_rounded =
              vcvtnq_s32_f32(vmulq_f32(vin0123, vinv_scale));
          const int32x4_t v4567_rounded =
              vcvtnq_s32_f32(vmulq_f32(vin4567, vinv_scale));
          const int16x8_t v01234567_packed = vqaddq_s16(
              vqmovn_high_s32(vqmovn_s32(v0123_rounded), v4567_rounded),
              vzero_point);
          const uint8x8_t vout01234567 = vqmovun_s16(v01234567_packed);
          vst1_u8(out, vout01234567);
          out += 8;
        }
        for (; e < elements_per_channel; ++e) {
          (*out++) = at::native::quantize_val_arm<uint8_t>(
              scales_data[c], zero_points_data[c], (*in++));
        }
      }
    }
  }
#endif // defined(__ARM_NEON__)
}
#endif // defined(__ARM_NEON__) || defined(__aarch64__)

void quantize_tensor_per_channel_affine_cpu(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  TORCH_CHECK(
      rtensor.is_contiguous() || (axis <= 1),
      "If tensor is channels_last contig then per channel quantization "
      "is supported only for axis = 0 or 1.");
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "quantize_tensor_per_channel_affine_cpu", [&]() {
        check_tensor_memory_format(rtensor, qtensor);
        quantize_tensor_per_channel_impl<scalar_t>(
            rtensor, qtensor, scales, zero_points, axis);
      });
}

template<typename T, typename N, typename Q>
void dequantize_per_channel_affine_kernel(
      const Tensor& qtensor,
      Tensor& rtensor,
      const Tensor& scales,
      const Tensor& zero_points,
      int64_t axis,
      int bit_width=8) {

  // For contiguous tensors, e.g. NCHW, arbitrary axis can be used.
  // For channels_last/3d however axis == 0 or 1.
  // Since current implementation on channels_last format does not
  // cover per channel quant with arbitrary axis value, it is better
  // to check and fail.
  TORCH_CHECK(rtensor.is_contiguous() || (axis <=1),
      "If tensor is channels_last contig then per channel quantization "
      "is supported only for axis = 0 or 1.");
  int64_t batches = size_to_dim_(axis, rtensor.sizes());
  int64_t elements_per_channel =
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      size_from_dim_(axis + 1, rtensor.sizes());
  int64_t channel = rtensor.size(axis);
  auto scales_data = scales.data_ptr<T>();
  auto zero_points_data = zero_points.data_ptr<N>();
  check_tensor_memory_format(qtensor, rtensor);
  const auto* qd = qtensor.const_data_ptr<Q>();
  float* rd = rtensor.data_ptr<float>();
  const auto elem_per_byte = 8 / bit_width;
  if (axis == 1 && (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
      rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
    for (const auto b : c10::irange(batches)) {
      for (const auto e : c10::irange(elements_per_channel)) {
        for (const auto c : c10::irange(channel)) {
          auto i = b * channel * elements_per_channel + e * channel + c;
          // We need to convert the qint8 value to float to ensure the
          // subtraction subexpression returns a float
          auto qvalue = qd[i / elem_per_byte].val_;
          if (bit_width < 8) {
            qvalue >>= (i % elem_per_byte) * bit_width;
            qvalue &= (1 << bit_width) - 1;
          }
          rd[i] = (static_cast<float>(qvalue) - zero_points_data[c]) * scales_data[c];
        }
      }
    }
  } else {
    for (const auto b : c10::irange(batches)) {
      for (const auto c : c10::irange(channel)) {
        for (const auto e : c10::irange(elements_per_channel)) {
          auto i = b * channel * elements_per_channel +
              c * elements_per_channel + e;
          // We need to convert the qint8 value to float to ensure the
          // subtraction subexpression returns a float
          auto qvalue = qd[i / elem_per_byte].val_;
          if (bit_width < 8) {
            qvalue >>= (i % elem_per_byte) * bit_width;
            qvalue &= (1 << bit_width) - 1;
          }
          rd[i] = (static_cast<float>(qvalue) - zero_points_data[c]) * scales_data[c];
        }
      }
    }
  }
}

void dequantize_tensor_per_channel_affine_cpu(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "dequantize_tensor_per_channel_affine_cpu", [&]() {
        dequantize_per_channel_affine_kernel<double, int64_t, scalar_t>(qtensor, rtensor, scales, zero_points, axis);
      });
}

// quantize stubs for floating point scale and zero_point.
void quantize_tensor_per_channel_float_qparams_cpu(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  // For contiguous tensors, e.g. NCHW, arbitrary axis can be used.
  // For channels_last/3d however axis == 0 or 1.
  // Since current implementation on channels_last format does not
  // cover per channel quant with arbitrary axis value, it is better
  // to check and fail.
  TORCH_CHECK(rtensor.is_contiguous() || (axis <=1),
      "If tensor is channels_last contig then per channel quantization "
      "is supported only for axis = 0 or 1.");
  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(
      qtensor.scalar_type(), "quantize_tensor_per_channel_float_qparams_cpu", [&]() {
        int64_t batches = size_to_dim_(axis, rtensor.sizes());
        int64_t elements_per_channel =
            size_from_dim_(axis + 1, rtensor.sizes());
        int64_t channel = rtensor.size(axis);
        auto scales_data = scales.data_ptr<float>();
        auto zero_points_data = zero_points.data_ptr<float>();
        check_tensor_memory_format(rtensor, qtensor);
        const float* rdata = rtensor.const_data_ptr<float>();
        auto qdata = reinterpret_cast<underlying_t*>(qtensor.data_ptr<scalar_t>());
        const auto elem_per_byte = CHAR_BIT / bit_width;
        int qvalue = 0;
        if (axis == 1 && (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
            rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
          for (const auto b : c10::irange(batches)) {
            for (const auto e : c10::irange(elements_per_channel)) {
              for (const auto c : c10::irange(channel)) {
                auto i = b * channel * elements_per_channel + e * channel + c;
                qvalue = quantize_val_float_qparams(
                    scales_data[c], zero_points_data[c], rdata[i], quant_min, quant_max);
                if (i % elem_per_byte == 0) {
                  qdata[i / elem_per_byte] = static_cast<underlying_t>(qvalue);
                } else {
                  qdata[i / elem_per_byte] |= static_cast<underlying_t>(qvalue << ((i % elem_per_byte) * bit_width));
                }
              }
            }
          }
        } else {
          for (const auto b : c10::irange(batches)) {
            for (const auto c : c10::irange(channel)) {
              for (const auto e : c10::irange(elements_per_channel)) {
                auto i = b * channel * elements_per_channel +
                    c * elements_per_channel + e;
                qvalue = quantize_val_float_qparams(
                    scales_data[c], zero_points_data[c], rdata[i], quant_min, quant_max);
                if (i % elem_per_byte == 0) {
                  qdata[i / elem_per_byte] = static_cast<underlying_t>(qvalue);
                } else {
                  qdata[i / elem_per_byte] |= static_cast<underlying_t>(qvalue << ((i % elem_per_byte) * bit_width));
                }
              }
            }
          }
        }
      });
}

void dequantize_tensor_per_channel_float_qparams_cpu(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(
      qtensor.scalar_type(), "dequantize_tensor_per_channel_float_qparams_cpu", [&]() {
        dequantize_per_channel_affine_kernel<float, float, scalar_t>(qtensor, rtensor, scales, zero_points, axis, bit_width);
      });
}

void quantize_tensor_per_tensor_affine_sub_byte_cpu(
    const Tensor& rtensor,
    Tensor& qtensor,
    float scale,
    float zero_point) {
  // TODO Use fbgemm kernel to pack values
  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(
    qtensor.scalar_type(), "quantize_tensor_per_tensor_affine_sub_byte_cpu", [&]() {
      check_tensor_memory_format(rtensor, qtensor);
      const float* const rdata = rtensor.const_data_ptr<float>();
      auto qdata = reinterpret_cast<underlying_t*>(qtensor.data_ptr<scalar_t>());
      auto numel = rtensor.numel();
      const auto elem_per_byte = CHAR_BIT / bit_width;
      for (const auto i : c10::irange(numel)) {
        float inv_scale = scale == 0 ? 1.0f : 1.0f / scale;
        int64_t qvalue = lrintf(std::nearbyint(rdata[i] * inv_scale) + zero_point);
        qvalue = std::max(quant_min, std::min(qvalue, quant_max));

        // We pack sub_byte values and align them to a byte.
        // Eg. for 4-bits Index 0 is packed in the lower 4-bits
        // and index 1 is packed in the upper 4-bits.
        if (i % elem_per_byte == 0) {
          qdata[i / elem_per_byte] = static_cast<underlying_t>(qvalue);
        } else {
          qdata[i / elem_per_byte] |= static_cast<underlying_t>(qvalue << ((i % elem_per_byte) * bit_width));
        }
      } // for numel
    });
}

void dequantize_tensor_per_tensor_affine_sub_byte_cpu(
    const Tensor& qtensor,
    Tensor& rtensor,
    float scale,
    float zero_point) {
  // TODO Use fbgemm kernel to pack values
  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(
    qtensor.scalar_type(), "dequantize_tensor_per_tensor_affine_sub_byte_cpu", [&]() {
      check_tensor_memory_format(rtensor, qtensor);
      auto rdata = rtensor.data_ptr<float>();
      const underlying_t* qdata = reinterpret_cast<const underlying_t*>(qtensor.const_data_ptr<scalar_t>());
      auto numel = rtensor.numel();
      const auto elem_per_byte = CHAR_BIT / bit_width;

      for (const auto i : c10::irange(numel)) {
        underlying_t qvalue = qdata[i / elem_per_byte];
        qvalue >>= (i % elem_per_byte) * bit_width;
        qvalue &= (1 << bit_width) - 1;
        rdata[i] = (static_cast<float>(qvalue) - zero_point) * scale;
      }
  });
}

// This function expects quantized_val input to already be quantized
template <typename scalar_t>
void cpu_masked_fill_kernel_quantized_cpu(TensorIterator& iter, scalar_t quantized_val) {
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    char* mask = data[1];
    for (const auto i : c10::irange(n)) {
      bool mask_value = *reinterpret_cast<bool*>(mask + strides[1] * i);

      if (mask_value) {
        *(scalar_t*)(dst + strides[0] * i) = quantized_val;
      }
    }
  };
  iter.for_each(loop);
}

void masked_fill_kernel_quantized_cpu(TensorIterator& iter, const Scalar& value, double scale, int zero_point) {
  AT_DISPATCH_QINT_TYPES(iter.dtype(), "masked_fill", [&] {
    float float_val = value.to<float>();
    auto quantized_val = quantize_val<scalar_t>(scale, zero_point, float_val);
    auto mask_dtype = iter.input_dtype(0);
    TORCH_CHECK(mask_dtype == ScalarType::Bool, "masked_fill only supports boolean masks, "
      "but got mask with dtype ", mask_dtype);
    cpu_masked_fill_kernel_quantized_cpu<scalar_t>(iter, quantized_val);
  });
}

// currently, we do not support accumulate=True for quantized tensors. We throw an exception in _index_put_impl_quantized_cpu_
void index_put_kernel_quantized_cpu(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate, double scale, int zero_point) {
  // NOTE: duplicate indices are only supported if accumulate is true.
  AT_DISPATCH_QINT_TYPES(iter.dtype(), "index_put", [&] {
    // See Note [Enabling Deterministic Operations]
    // Parallel cpu_index_kernel with accumulation is nondeterministic, so we
    // must enable serial execution if deterministic algorithms are enabled.
    const bool is_deterministic = at::globalContext().deterministicAlgorithms();
    at::native::cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [scale, zero_point](char* dst, char* src, int64_t offset) {
      *(scalar_t*)(dst + offset) = quantize_val<scalar_t>(scale, zero_point, *(float*)src);
    }, /*serial_execution=*/is_deterministic);
  });
}

template<typename T>
void _qmul_tensor_cpu_impl(
    T* out_ptr,
    int64_t size,
    const uint8_t* x_ptr,
    double x_scale,
    int64_t x_zero_point,
    const uint8_t* y_ptr,
    double y_scale,
    int64_t y_zero_point,
    double output_scale,
    int64_t output_zero_point) {
  float multiplier = x_scale * y_scale / output_scale;
  auto compute_with_scalar = [&](int idx) {
    uint8_t x_data = *(x_ptr + idx);
    uint8_t y_data = *(y_ptr + idx);
    int32_t x_val = static_cast<int32_t>(x_data) - x_zero_point;
    int32_t y_val = static_cast<int32_t>(y_data) - y_zero_point;
    int32_t out_val = x_val * y_val;
    float out_val_f = (float)out_val * multiplier;
    if constexpr (std::is_same<T, float>::value) {
      *(out_ptr + idx) = out_val_f;
    } else if constexpr (std::is_same<T, at::BFloat16>::value) {
      *(out_ptr + idx) = at::BFloat16(out_val_f);
    } else if constexpr (std::is_same<T, at::Half>::value) {
      *(out_ptr + idx) = at::Half(out_val_f);
    } else { //  T == uint8, requantization needed
      out_val_f = std::round(out_val_f);
      int32_t out_val_i32 = (int32_t)out_val_f + output_zero_point;
      out_val_i32 = std::min(255, std::max(0, out_val_i32));
      *(out_ptr + idx) = static_cast<uint8_t>(out_val_i32);
    }
  };
#if defined(CPU_CAPABILITY_AVX512)
  int64_t size_rem = size % 16;
  int64_t size_com = size - size_rem;
  int64_t steps = size_com / 16;
  __m512 vs = _mm512_set1_ps(multiplier);
  __m512i vza = _mm512_set1_epi32(x_zero_point);
  __m512i vzb = _mm512_set1_epi32(y_zero_point);
  __m512i vzc = _mm512_set1_epi32(output_zero_point);
  __m512i v255 = _mm512_set1_epi32(255);
  __m512i v0 = _mm512_set1_epi32(0);
  at::parallel_for(0, steps, 1, [&](int64_t start, int64_t end) {
    for (const auto d : c10::irange(start, end)) {
      auto x_data = x_ptr + d * 16;
      auto y_data = y_ptr + d * 16;
      auto out_data = out_ptr + d * 16;
      __m128i va = _mm_loadu_si128((__m128i*)x_data);
      __m128i vb = _mm_loadu_si128((__m128i*)y_data);
      __m512i va_i32 = _mm512_cvtepi8_epi32(va);
      __m512i vb_i32 = _mm512_cvtepi8_epi32(vb);
      va_i32 = _mm512_sub_epi32(va_i32, vza);
      vb_i32 = _mm512_sub_epi32(vb_i32, vzb);
      __m512i vc = _mm512_mullo_epi32(va_i32, vb_i32);
      __m512 vc_f = _mm512_cvtepi32_ps(vc);
      vc_f = _mm512_mul_ps(vc_f, vs);
      if constexpr (std::is_same<T, float>::value) {
        _mm512_storeu_ps(out_data, vc_f);
      } else if constexpr (std::is_same<T, at::BFloat16>::value) {
        __m256i vc_bf16 = cvtfp32_bf16(vc_f);
        _mm256_storeu_si256((__m256i*)out_data, vc_bf16);
      } else if constexpr (std::is_same<T, at::Half>::value) {
        __m256i vc_f16 = cvtfp32_fp16(vc_f);
        _mm256_storeu_si256((__m256i*)out_data, vc_f16);
      } else { //  T == uint8, requantization needed
        __m512i vc_i32 = _mm512_cvtps_epi32(vc_f);
        vc_i32 = _mm512_add_epi32(vc_i32, vzc);
        vc_i32 = _mm512_min_epi32(vc_i32, v255);
        vc_i32 = _mm512_max_epi32(vc_i32, v0);
        __m128i vc_i8 = _mm512_cvtepi32_epi8(vc_i32);
        _mm_storeu_si128((__m128i*)out_data, vc_i8);
      }
    }
  });
  if (size_rem > 0) {
    for (const auto d : c10::irange(size_rem)) {
      compute_with_scalar(size_com + d);
    }
  }
#else
  at::parallel_for(0, size, 1, [&](int64_t start, int64_t end) {
    for (const auto d : c10::irange(start, end)) {
      compute_with_scalar(d);
    }
  });
#endif
}

void qmul_tensor_cpu_kernel(
    Tensor& out,
    const Tensor& qx,
    double qx_scale,
    int64_t qx_zero_point,
    const Tensor& qy,
    double qy_scale,
    int64_t qy_zero_point,
    double output_scale,
    int64_t output_zero_point) {
  auto qx_ptr = qx.const_data_ptr<uint8_t>();
  auto qy_ptr = qy.const_data_ptr<uint8_t>();
  int64_t size = qx.numel();
  TORCH_CHECK(
      size == qy.numel() && size == out.numel(),
      "qmul_cpu: Expect qx, qy and out to have the same number of elements");
  AT_DISPATCH_FLOATING_TYPES_AND3(
      at::ScalarType::BFloat16, at::ScalarType::Half, at::ScalarType::Byte, out.scalar_type(), "int8_mul_cpu", [&] {
        auto out_ptr = out.data_ptr<scalar_t>();
        _qmul_tensor_cpu_impl<scalar_t>(
            out_ptr, size, qx_ptr, qx_scale, qx_zero_point, qy_ptr, qy_scale, qy_zero_point, output_scale, output_zero_point);
      });
}

template<typename T, bool ReLUFused>
void _qadd_tensor_cpu_impl(
    T* out_ptr,
    int64_t size,
    const uint8_t* x_ptr,
    double x_scale,
    int64_t x_zero_point,
    const uint8_t* y_ptr,
    double y_scale,
    int64_t y_zero_point,
    double output_scale,
    int64_t output_zero_point) {
  float inv_output_scale = 1.0 / output_scale;
  auto compute_with_scalar = [&](int idx) {
    uint8_t x_data = *(x_ptr + idx);
    uint8_t y_data = *(y_ptr + idx);
    int32_t x_val = static_cast<int32_t>(x_data) - x_zero_point;
    int32_t y_val = static_cast<int32_t>(y_data) - y_zero_point;
    float x_val_f = static_cast<float>(x_val) * x_scale;
    float y_val_f = static_cast<float>(y_val) * y_scale;
    float out_val_f = x_val_f + y_val_f;
    if constexpr (ReLUFused) {
      out_val_f = std::max(out_val_f, 0.f);
    }
    if constexpr (std::is_same<T, float>::value) {
      *(out_ptr + idx) = out_val_f;
    } else if constexpr (std::is_same<T, at::BFloat16>::value) {
      *(out_ptr + idx) = at::BFloat16(out_val_f);
    } else if constexpr (std::is_same<T, at::Half>::value) {
      *(out_ptr + idx) = at::Half(out_val_f);
    } else { //  T == uint8, requantization needed
      out_val_f = std::round(out_val_f * inv_output_scale);
      int32_t out_val_i32 = (int32_t)out_val_f + output_zero_point;
      out_val_i32 = std::min(255, std::max(0, out_val_i32));
      *(out_ptr + idx) = static_cast<uint8_t>(out_val_i32);
    }
  };
#if defined(CPU_CAPABILITY_AVX512)
  int64_t size_rem = size % 16;
  int64_t size_com = size - size_rem;
  int64_t steps = size_com / 16;
  __m512 vsa = _mm512_set1_ps(x_scale);
  __m512 vsb = _mm512_set1_ps(y_scale);
  __m512 vsc = _mm512_set1_ps(inv_output_scale);
  __m512i vza = _mm512_set1_epi32(x_zero_point);
  __m512i vzb = _mm512_set1_epi32(y_zero_point);
  __m512i vzc = _mm512_set1_epi32(output_zero_point);
  __m512i v255 = _mm512_set1_epi32(255);
  __m512i v0 = _mm512_set1_epi32(0);
  __m512 v0f = _mm512_set1_ps(0);
  at::parallel_for(0, steps, 1, [&](int64_t start, int64_t end) {
    for (const auto d : c10::irange(start, end)) {
      auto x_data = x_ptr + d * 16;
      auto y_data = y_ptr + d * 16;
      auto out_data = out_ptr + d * 16;
      __m128i va = _mm_loadu_si128((__m128i*)x_data);
      __m128i vb = _mm_loadu_si128((__m128i*)y_data);
      __m512i va_i32 = _mm512_cvtepi8_epi32(va);
      __m512i vb_i32 = _mm512_cvtepi8_epi32(vb);
      va_i32 = _mm512_sub_epi32(va_i32, vza);
      vb_i32 = _mm512_sub_epi32(vb_i32, vzb);
      __m512 va_f = _mm512_cvtepi32_ps(va_i32);
      __m512 vb_f = _mm512_cvtepi32_ps(vb_i32);
      va_f = _mm512_mul_ps(va_f, vsa);
      vb_f = _mm512_mul_ps(vb_f, vsb);
      __m512 vc_f = _mm512_add_ps(va_f, vb_f);
      if constexpr (ReLUFused) {
        vc_f = _mm512_max_ps(vc_f, v0f);
      }
      if constexpr (std::is_same<T, float>::value) {
        _mm512_storeu_ps(out_data, vc_f);
      } else if constexpr (std::is_same<T, at::BFloat16>::value) {
        __m256i vc_bf16 = cvtfp32_bf16(vc_f);
        _mm256_storeu_si256((__m256i*)out_data, vc_bf16);
      } else if constexpr (std::is_same<T, at::Half>::value) {
        __m256i vc_f16 = cvtfp32_fp16(vc_f);
        _mm256_storeu_si256((__m256i*)out_data, vc_f16);
      } else { //  T == uint8, requantization needed
        vc_f = _mm512_mul_ps(vc_f, vsc);
        __m512i vc_i32 = _mm512_cvtps_epi32(vc_f);
        vc_i32 = _mm512_add_epi32(vc_i32, vzc);
        vc_i32 = _mm512_min_epi32(vc_i32, v255);
        vc_i32 = _mm512_max_epi32(vc_i32, v0);
        __m128i vc_i8 = _mm512_cvtepi32_epi8(vc_i32);
        _mm_storeu_si128((__m128i*)out_data, vc_i8);
      }
    }
  });
  if (size_rem > 0) {
    for (const auto d : c10::irange(size_rem)) {
      compute_with_scalar(size_com + d);
    }
  }
#else
  at::parallel_for(0, size, 1, [&](int64_t start, int64_t end) {
    for (const auto d : c10::irange(start, end)) {
      compute_with_scalar(d);
    }
  });
#endif
}

template <bool ReLUFused>
void qadd_tensor_cpu_kernel(
    Tensor& out,
    const Tensor& qx,
    double qx_scale,
    int64_t qx_zero_point,
    const Tensor& qy,
    double qy_scale,
    int64_t qy_zero_point,
    double output_scale,
    int64_t output_zero_point) {
  auto qx_ptr = qx.const_data_ptr<uint8_t>();
  auto qy_ptr = qy.const_data_ptr<uint8_t>();
  int64_t size = qx.numel();
  TORCH_CHECK(
      size == qy.numel() && size == out.numel(),
      "qadd_cpu: Expect qx, qy and out to have the same number of elements");
  AT_DISPATCH_FLOATING_TYPES_AND3(
      at::ScalarType::BFloat16, at::ScalarType::Half, at::ScalarType::Byte, out.scalar_type(), "int8_add_cpu", [&] {
        auto out_ptr = out.data_ptr<scalar_t>();
        _qadd_tensor_cpu_impl<scalar_t, ReLUFused>(
            out_ptr, size, qx_ptr, qx_scale, qx_zero_point, qy_ptr, qy_scale, qy_zero_point, output_scale, output_zero_point);
      });
}
} // anonymous namespace

// Some quantization tests are flaky on Windows with AVX512. If --continue-through-error
// is used, only one fails. But if the failing test is skipped, another one fails.
// If the second test is also skipped, a third one fails.
// So, until Quantization support for Windows is fixed for AVX512,
// AVX2 kernels would be used instead. Ref: GH 56992.
#if defined(_WIN32)
REGISTER_DISPATCH(dequantize_tensor_per_channel_affine_stub,
                  &dequantize_tensor_per_channel_affine_cpu)
REGISTER_DISPATCH(dequantize_tensor_per_channel_float_qparams_stub,
                  &dequantize_tensor_per_channel_float_qparams_cpu)
REGISTER_DISPATCH(fake_quant_per_channel_cachemask_stub,
                  &fake_quant_per_channel_cachemask_cpu)
REGISTER_DISPATCH(qavg_pool2d_nhwc_stub, &qavg_pool2d_nhwc_kernel)
REGISTER_DISPATCH(qavg_pool3d_nhwc_stub, &qavg_pool3d_nhwc_kernel)
#else
// These kernels are dispatched to AVX512
ALSO_REGISTER_AVX512_DISPATCH(dequantize_tensor_per_channel_affine_stub,
                  &dequantize_tensor_per_channel_affine_cpu)
ALSO_REGISTER_AVX512_DISPATCH(dequantize_tensor_per_channel_float_qparams_stub,
                  &dequantize_tensor_per_channel_float_qparams_cpu)
ALSO_REGISTER_AVX512_DISPATCH(fake_quant_per_channel_cachemask_stub,
                  &fake_quant_per_channel_cachemask_cpu)
ALSO_REGISTER_AVX512_DISPATCH(qavg_pool2d_nhwc_stub, &qavg_pool2d_nhwc_kernel)
ALSO_REGISTER_AVX512_DISPATCH(qavg_pool3d_nhwc_stub, &qavg_pool3d_nhwc_kernel)
#endif // CPU_CAPABILITY_AVX512 && _WIN32

// The kernels below are dispatched to AVX2 because they don't perform as well
// with AVX512. We might revisit this decision in the near future.
REGISTER_DISPATCH(dequantize_tensor_per_tensor_affine_stub,
                  &dequantize_tensor_per_tensor_affine_cpu)
REGISTER_DISPATCH(fake_quant_grad_learnable_tensor_stub,
                  &fake_quantize_learnable_tensor_grad_kernel_cpu)
REGISTER_DISPATCH(fake_quant_tensor_cachemask_stub,
                  &fake_quantize_tensor_cachemask_kernel)
REGISTER_DISPATCH(fake_quant_tensor_cachemask_tensor_qparams_stub,
                  &fake_quantize_tensor_cachemask_tensor_qparams_kernel)
REGISTER_DISPATCH(qadaptive_avg_pool2d_nhwc_stub,
                  &qadaptive_avg_pool2d_nhwc_kernel)
REGISTER_DISPATCH(qadaptive_avg_pool3d_ndhwc_stub,
                  &qadaptive_avg_pool3d_ndhwc_kernel)
REGISTER_DISPATCH(qadd_relu_stub, &qadd_kernel<true>)
REGISTER_DISPATCH(qadd_scalar_relu_stub, &qadd_scalar_kernel<true>)
REGISTER_DISPATCH(qadd_scalar_stub, &qadd_scalar_kernel<false>)
REGISTER_DISPATCH(qadd_stub, &qadd_kernel<false>)

REGISTER_DISPATCH(qbatch_norm_relu_stub, &q_batch_norm_kernel<true>)
REGISTER_DISPATCH(qbatch_norm_stub, &q_batch_norm_kernel<false>)
REGISTER_DISPATCH(qcat_nhwc_stub, &qcat_nhwc_kernel<false>)
REGISTER_DISPATCH(qcat_relu_nhwc_stub, &qcat_nhwc_kernel<true>)
REGISTER_DISPATCH(qclamp_stub, &qclamp_kernel)
REGISTER_DISPATCH(qclamp_min_stub, &qclamp_min_kernel)
REGISTER_DISPATCH(qclamp_max_stub, &qclamp_max_kernel)
REGISTER_DISPATCH(qelu_stub, &qelu_kernel)
REGISTER_DISPATCH(qhardsigmoid_stub, &qhardsigmoid_kernel)
REGISTER_DISPATCH(qhardswish_stub, &qhardswish_kernel)
REGISTER_DISPATCH(qmaxpool_2d_nhwc_stub, &qmaxpool_2d_nhwc_kernel)
REGISTER_DISPATCH(qmaxpool_3d_nthwc_stub, &qmaxpool_3d_nthwc_kernel)
REGISTER_DISPATCH(qmul_relu_stub, &qmul_kernel<true>)
REGISTER_DISPATCH(qmul_stub, &qmul_kernel<false>)
REGISTER_DISPATCH(qrelu_leaky_stub, &leaky_qrelu_out_kernel)
REGISTER_DISPATCH(qrelu_stub, &qrelu_kernel)
REGISTER_DISPATCH(qprelu_stub, &qprelu_out_kernel)
REGISTER_DISPATCH(qgelu_stub, &qgelu_kernel)
REGISTER_DISPATCH(qsigmoid_stub, &qsigmoid_kernel)
REGISTER_DISPATCH(qtanh_stub, &qtanh_kernel)
REGISTER_DISPATCH(qthreshold_stub, &qthreshold_kernel)
REGISTER_DISPATCH(qtopk_stub, &qtopk_kernel)
REGISTER_DISPATCH(fake_quant_grad_learnable_channel_stub,
                  &fake_quantize_learnable_channel_grad_kernel_cpu)
REGISTER_DISPATCH(
    quantize_tensor_per_tensor_affine_stub,
    &quantize_tensor_per_tensor_affine_cpu)
REGISTER_DISPATCH(
    quantize_tensor_per_channel_affine_stub,
    &quantize_tensor_per_channel_affine_cpu)
REGISTER_DISPATCH(
    quantize_tensor_per_channel_float_qparams_stub,
    &quantize_tensor_per_channel_float_qparams_cpu)
REGISTER_DISPATCH(quantized_normalize_stub, &quantized_normalize_kernel)
REGISTER_DISPATCH(quantized_groupnorm_nhwc_stub, &quantized_groupnorm_nhwc_kernel)
REGISTER_DISPATCH(qupsample_bilinear2d_nhwc_stub,
                  &qupsample_bilinear2d_nhwc_kernel)
REGISTER_DISPATCH(
    quantize_tensor_per_tensor_affine_sub_byte_stub,
    &quantize_tensor_per_tensor_affine_sub_byte_cpu)
REGISTER_DISPATCH(
    dequantize_tensor_per_tensor_affine_sub_byte_stub,
    &dequantize_tensor_per_tensor_affine_sub_byte_cpu)
REGISTER_DISPATCH(
    masked_fill_kernel_quantized_stub,
    &masked_fill_kernel_quantized_cpu)
REGISTER_DISPATCH(
    index_put_kernel_quantized_stub,
    &index_put_kernel_quantized_cpu)
REGISTER_DISPATCH(qmean_inner_dim_stub, &qmean_inner_dim_kernel)
REGISTER_DISPATCH(qstd_inner_dim_stub, &qstd_inner_dim_kernel)
ALSO_REGISTER_AVX512_DISPATCH(qmul_tensor_cpu_stub, &qmul_tensor_cpu_kernel)
ALSO_REGISTER_AVX512_DISPATCH(qadd_tensor_cpu_stub, &qadd_tensor_cpu_kernel<false>)
ALSO_REGISTER_AVX512_DISPATCH(qadd_relu_tensor_cpu_stub, &qadd_tensor_cpu_kernel<true>)
ALSO_REGISTER_AVX512_DISPATCH(qbatch_norm_cpu_stub, &q_batch_norm_cpu_kernel)
} // namespace at::native
// NOLINTEND(*-c-arrays)
