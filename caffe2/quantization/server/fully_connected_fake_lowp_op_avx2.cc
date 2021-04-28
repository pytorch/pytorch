#include <immintrin.h>

#include <c10/util/irange.h>

namespace caffe2 {

namespace {

// NOTE: clang-format wants to use a different formatting but the
// current formatting should be easier to read.
// clang-format off
alignas(64) const int ld_st_masks[8][8] = {
  {  0,  0,  0,  0,  0,  0,  0,  0, },
  { -1,  0,  0,  0,  0,  0,  0,  0, },
  { -1, -1,  0,  0,  0,  0,  0,  0, },
  { -1, -1, -1,  0,  0,  0,  0,  0, },
  { -1, -1, -1, -1,  0,  0,  0,  0, },
  { -1, -1, -1, -1, -1,  0,  0,  0, },
  { -1, -1, -1, -1, -1, -1,  0,  0, },
  { -1, -1, -1, -1, -1, -1, -1,  0, },
};
// clang-format on

} // anonymous namespace

// convert to float16 reducing mantissa, preserving exponent
void fp32_to_bfp16(const float* source, size_t size, float* dest) {
  // Results on a 1 sign, 8 exponent, 7 mantissa
  constexpr int mask = 0xFFFF0000;
  __m256 wmask = _mm256_broadcast_ss(reinterpret_cast<const float*>(&mask));

  size_t i = 0;
  for (; i < (size / 8) * 8; i += 8) {
    __m256 data = _mm256_loadu_ps(&source[i]);
    _mm256_storeu_ps(&dest[i], _mm256_and_ps(wmask, data));
  }
  if (i < size) {
    __m256i ld_st_mask = _mm256_load_si256(
        reinterpret_cast<const __m256i*>(ld_st_masks[size - i]));
    __m256 data = _mm256_maskload_ps(&source[i], ld_st_mask);
    _mm256_maskstore_ps(&dest[i], ld_st_mask, _mm256_and_ps(wmask, data));
  }
}

// convert to float24 reducing mantissa, preserving exponent
void fp32_to_bfp24(const float* source, size_t size, float* dest) {
  // Results on a 1 sign, 8 exponent, 7 mantissa
  constexpr int mask = 0xFFFFFF00;
  __m256 wmask = _mm256_broadcast_ss(reinterpret_cast<const float*>(&mask));

  size_t i = 0;
  for (; i < (size / 8) * 8; i += 8) {
    __m256 data = _mm256_loadu_ps(&source[i]);
    _mm256_storeu_ps(&dest[i], _mm256_and_ps(wmask, data));
  }
  if (i < size) {
    __m256i ld_st_mask = _mm256_load_si256(
        reinterpret_cast<const __m256i*>(ld_st_masks[size - i]));
    __m256 data = _mm256_maskload_ps(&source[i], ld_st_mask);
    _mm256_maskstore_ps(&dest[i], ld_st_mask, _mm256_and_ps(wmask, data));
  }
}

// convert to float14 reducing mantissa, preserving exponent
void fp32_to_bfp14(const float* source, size_t size, float* dest) {
  // Results on a 1 sign, 8 exponent, 7 mantissa
  constexpr int mask = 0xFFFC0000;
  __m256 wmask = _mm256_broadcast_ss((float*)(&mask));

  size_t i = 0;
  for (; i < (size / 8) * 8; i += 8) {
    __m256 data = _mm256_loadu_ps(&source[i]);
    _mm256_storeu_ps(&dest[i], _mm256_and_ps(wmask, data));
  }
  if (i < size) {
    __m256i ld_st_mask = _mm256_load_si256(
        reinterpret_cast<const __m256i*>(ld_st_masks[size - i]));
    __m256 data = _mm256_maskload_ps(&source[i], ld_st_mask);
    _mm256_maskstore_ps(&dest[i], ld_st_mask, _mm256_and_ps(wmask, data));
  }
}

void fp32_to_bfp16_scalar(const float* source, size_t size, float* dest) {
  constexpr int mask = 0xFFFF0000;
  for (const auto i : c10::irange(size)) {
    *(int*)(dest + i) = *(int*)(source + i) & mask;
  }
}

// convert to IEEE float16
void fp32_to_fp16(const float* source, size_t size, float* dest) {
  size_t i = 0;
  for (; i < (size / 8) * 8; i += 8) {
    __m128i vin_fp16 = _mm256_cvtps_ph(_mm256_loadu_ps(&source[i]), 0);
    _mm256_storeu_ps(&dest[i], _mm256_cvtph_ps(vin_fp16));
  }
  if (i < size) {
    __m256i ld_st_mask = _mm256_load_si256(
        reinterpret_cast<const __m256i*>(ld_st_masks[size - i]));
    __m128i vin_fp16 =
        _mm256_cvtps_ph(_mm256_maskload_ps(&source[i], ld_st_mask), 0);
    _mm256_maskstore_ps(&dest[i], ld_st_mask, _mm256_cvtph_ps(vin_fp16));
  }
}

// fp32 -> int32 -> += 1<< 15 -> fp32 -> truncation
void fp32_to_bfp16_round(const float* source, size_t size, float* dest) {
  constexpr int offset = 0x00008000; // 1 << 15
  constexpr int mask = 0xFFFF0000;

  __m256i woffset = _mm256_set1_epi32(offset);
  __m256i wmask = _mm256_set1_epi32(mask);

  size_t i = 0;
  for (; i < (size / 8) * 8; i += 8) {
    __m256i v32int = _mm256_add_epi32(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&source[i])),
        woffset);
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(&dest[i]), _mm256_and_si256(wmask, v32int));
  }
  if (i < size) {
    __m256i ld_st_mask = _mm256_load_si256(
        reinterpret_cast<const __m256i*>(ld_st_masks[size - i]));
    __m256i v32int = _mm256_add_epi32(
        _mm256_maskload_epi32(
            reinterpret_cast<const int*>(&source[i]), ld_st_mask),
        woffset);
    _mm256_maskstore_epi32(
        reinterpret_cast<int*>(&dest[i]),
        ld_st_mask,
        _mm256_and_si256(wmask, v32int));
  }
}

} // namespace caffe2
