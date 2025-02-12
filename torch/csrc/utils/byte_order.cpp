#include <c10/util/BFloat16.h>
#include <c10/util/complex.h>
#include <c10/util/irange.h>
#include <torch/csrc/utils/byte_order.h>

#include <cstring>
#include <vector>

#if defined(_MSC_VER)
#include <stdlib.h>
#endif
namespace {

static void swapBytes16(void* ptr) {
  uint16_t output = 0;
  memcpy(&output, ptr, sizeof(uint16_t));
#if defined(_MSC_VER) && !defined(_DEBUG)
  output = _byteswap_ushort(output);
#elif defined(__llvm__) || defined(__GNUC__) && !defined(__ICC)
  output = __builtin_bswap16(output);
#else
  uint16_t Hi = output >> 8;
  uint16_t Lo = output << 8;
  output = Hi | Lo;
#endif
  memcpy(ptr, &output, sizeof(uint16_t));
}

static void swapBytes32(void* ptr) {
  uint32_t output = 0;
  memcpy(&output, ptr, sizeof(uint32_t));
#if defined(_MSC_VER) && !defined(_DEBUG)
  output = _byteswap_ulong(output);
#elif defined(__llvm__) || defined(__GNUC__) && !defined(__ICC)
  output = __builtin_bswap32(output);
#else
  uint32_t Byte0 = output & 0x000000FF;
  uint32_t Byte1 = output & 0x0000FF00;
  uint32_t Byte2 = output & 0x00FF0000;
  uint32_t Byte3 = output & 0xFF000000;
  output = (Byte0 << 24) | (Byte1 << 8) | (Byte2 >> 8) | (Byte3 >> 24);
#endif
  memcpy(ptr, &output, sizeof(uint32_t));
}

static void swapBytes64(void* ptr) {
  uint64_t output = 0;
  memcpy(&output, ptr, sizeof(uint64_t));
#if defined(_MSC_VER)
  output = _byteswap_uint64(output);
#elif defined(__llvm__) || defined(__GNUC__) && !defined(__ICC)
  output = __builtin_bswap64(output);
#else
  uint64_t Byte0 = output & 0x00000000000000FF;
  uint64_t Byte1 = output & 0x000000000000FF00;
  uint64_t Byte2 = output & 0x0000000000FF0000;
  uint64_t Byte3 = output & 0x00000000FF000000;
  uint64_t Byte4 = output & 0x000000FF00000000;
  uint64_t Byte5 = output & 0x0000FF0000000000;
  uint64_t Byte6 = output & 0x00FF000000000000;
  uint64_t Byte7 = output & 0xFF00000000000000;
  output = (Byte0 << (7 * 8)) | (Byte1 << (5 * 8)) | (Byte2 << (3 * 8)) |
      (Byte3 << (1 * 8)) | (Byte7 >> (7 * 8)) | (Byte6 >> (5 * 8)) |
      (Byte5 >> (3 * 8)) | (Byte4 >> (1 * 8));
#endif
  memcpy(ptr, &output, sizeof(uint64_t));
}

static uint16_t decodeUInt16(const uint8_t* data) {
  uint16_t output = 0;
  memcpy(&output, data, sizeof(uint16_t));
  return output;
}

static uint16_t decodeUInt16ByteSwapped(const uint8_t* data) {
  uint16_t output = decodeUInt16(data);
  swapBytes16(&output);
  return output;
}

static uint32_t decodeUInt32(const uint8_t* data) {
  uint32_t output = 0;
  memcpy(&output, data, sizeof(uint32_t));
  return output;
}

static uint32_t decodeUInt32ByteSwapped(const uint8_t* data) {
  uint32_t output = decodeUInt32(data);
  swapBytes32(&output);
  return output;
}

static uint64_t decodeUInt64(const uint8_t* data) {
  uint64_t output = 0;
  memcpy(&output, data, sizeof(uint64_t));
  return output;
}

static uint64_t decodeUInt64ByteSwapped(const uint8_t* data) {
  uint64_t output = decodeUInt64(data);
  swapBytes64(&output);
  return output;
}

} // anonymous namespace

namespace torch::utils {

THPByteOrder THP_nativeByteOrder() {
  uint32_t x = 1;
  return *(uint8_t*)&x ? THP_LITTLE_ENDIAN : THP_BIG_ENDIAN;
}

template <typename T, typename U>
void THP_decodeBuffer(T* dst, const uint8_t* src, U type, size_t len) {
  if constexpr (std::is_same_v<U, THPByteOrder>)
    THP_decodeBuffer(dst, src, type != THP_nativeByteOrder(), len);
  else {
    auto func = [&](const uint8_t* src_data) {
      if constexpr (std::is_same_v<T, int16_t>) {
        return type ? decodeUInt16ByteSwapped(src_data)
                    : decodeUInt16(src_data);
      } else if constexpr (std::is_same_v<T, int32_t>) {
        return type ? decodeUInt32ByteSwapped(src_data)
                    : decodeUInt32(src_data);
      } else if constexpr (std::is_same_v<T, int64_t>) {
        return type ? decodeUInt64ByteSwapped(src_data)
                    : decodeUInt64(src_data);
      }
    };

    for (const auto i : c10::irange(len)) {
      dst[i] = static_cast<T>(func(src));
      src += sizeof(T);
    }
  }
}

template <>
TORCH_API void THP_decodeBuffer<c10::Half, bool>(
    c10::Half* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint16_t x;
      c10::Half f;
    };
    x = (do_byte_swap ? decodeUInt16ByteSwapped(src) : decodeUInt16(src));
    dst[i] = f;
    src += sizeof(uint16_t);
  }
}

template <>
TORCH_API void THP_decodeBuffer<at::BFloat16, bool>(
    at::BFloat16* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    uint16_t x =
        (do_byte_swap ? decodeUInt16ByteSwapped(src) : decodeUInt16(src));
    std::memcpy(&dst[i], &x, sizeof(dst[i]));
    src += sizeof(uint16_t);
  }
}

template <>
TORCH_API void THP_decodeBuffer<bool, bool>(
    bool* dst,
    const uint8_t* src,
    bool,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    dst[i] = (int)src[i] != 0 ? true : false;
  }
}

template <>
TORCH_API void THP_decodeBuffer<float, bool>(
    float* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint32_t x;
      float f;
    };
    x = (do_byte_swap ? decodeUInt32ByteSwapped(src) : decodeUInt32(src));
    dst[i] = f;
    src += sizeof(float);
  }
}

template <>
TORCH_API void THP_decodeBuffer<double, bool>(
    double* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint64_t x;
      double d;
    };
    x = (do_byte_swap ? decodeUInt64ByteSwapped(src) : decodeUInt64(src));
    dst[i] = d;
    src += sizeof(double);
  }
}

template <>
TORCH_API void THP_decodeBuffer<c10::complex<float>, bool>(
    c10::complex<float>* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint32_t x;
      float re;
    };
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint32_t y;
      float im;
    };

    x = (do_byte_swap ? decodeUInt32ByteSwapped(src) : decodeUInt32(src));
    src += sizeof(float);
    y = (do_byte_swap ? decodeUInt32ByteSwapped(src) : decodeUInt32(src));
    src += sizeof(float);

    dst[i] = c10::complex<float>(re, im);
  }
}

template <>
TORCH_API void THP_decodeBuffer<c10::complex<double>, bool>(
    c10::complex<double>* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint64_t x;
      double re;
    };
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint64_t y;
      double im;
    };
    static_assert(sizeof(uint64_t) == sizeof(double));

    x = (do_byte_swap ? decodeUInt64ByteSwapped(src) : decodeUInt64(src));
    src += sizeof(double);
    y = (do_byte_swap ? decodeUInt64ByteSwapped(src) : decodeUInt64(src));
    src += sizeof(double);

    dst[i] = c10::complex<double>(re, im);
  }
}

#define DEFINE_DECODE(TYPE, ORDER)                       \
  template TORCH_API void THP_decodeBuffer<TYPE, ORDER>( \
      TYPE * dst, const uint8_t* src, ORDER type, size_t len);

DEFINE_DECODE(int16_t, THPByteOrder)
DEFINE_DECODE(int32_t, THPByteOrder)
DEFINE_DECODE(int64_t, THPByteOrder)
DEFINE_DECODE(c10::Half, THPByteOrder)
DEFINE_DECODE(float, THPByteOrder)
DEFINE_DECODE(double, THPByteOrder)
DEFINE_DECODE(c10::BFloat16, THPByteOrder)
DEFINE_DECODE(c10::complex<float>, THPByteOrder)
DEFINE_DECODE(c10::complex<double>, THPByteOrder)

DEFINE_DECODE(int16_t, bool)
DEFINE_DECODE(int32_t, bool)
DEFINE_DECODE(int64_t, bool)

#undef DEFINE_DECODE

template <typename T>
void THP_encodeBuffer(
    uint8_t* dst,
    const T* src,
    THPByteOrder order,
    size_t len) {
  memcpy(dst, src, sizeof(T) * len);
  if (order != THP_nativeByteOrder()) {
    for (const auto i : c10::irange(len)) {
      (void)i;
      if constexpr (std::is_same_v<T, int16_t>) {
        swapBytes16(dst);
      } else if constexpr (
          std::is_same_v<T, int32_t> || std::is_same_v<T, float>) {
        swapBytes32(dst);
      } else if constexpr (
          std::is_same_v<T, int64_t> || std::is_same_v<T, double>) {
        swapBytes64(dst);
      }
      dst += sizeof(T);
    }
  }
}

template <typename T>
static std::vector<T> complex_to_float(const c10::complex<T>* src, size_t len) {
  std::vector<T> new_src;
  new_src.reserve(2 * len);
  for (const auto i : c10::irange(len)) {
    auto elem = src[i];
    new_src.emplace_back(elem.real());
    new_src.emplace_back(elem.imag());
  }
  return new_src;
}

template <>
TORCH_API void THP_encodeBuffer<c10::complex<float>>(
    uint8_t* dst,
    const c10::complex<float>* src,
    THPByteOrder order,
    size_t len) {
  auto new_src = complex_to_float(src, len);
  memcpy(dst, static_cast<void*>(&new_src), 2 * sizeof(float) * len);
  if (order != THP_nativeByteOrder()) {
    for (const auto i : c10::irange(2 * len)) {
      (void)i; // Suppress unused variable warning
      swapBytes32(dst);
      dst += sizeof(float);
    }
  }
}

template <>
TORCH_API void THP_encodeBuffer<c10::complex<double>>(
    uint8_t* dst,
    const c10::complex<double>* src,
    THPByteOrder order,
    size_t len) {
  auto new_src = complex_to_float(src, len);
  memcpy(dst, static_cast<void*>(&new_src), 2 * sizeof(double) * len);
  if (order != THP_nativeByteOrder()) {
    for (const auto i : c10::irange(2 * len)) {
      (void)i; // Suppress unused variable warning
      swapBytes64(dst);
      dst += sizeof(double);
    }
  }
}

#define DEFINE_ENCODE(TYPE)                       \
  template TORCH_API void THP_encodeBuffer<TYPE>( \
      uint8_t * dst, const TYPE* src, THPByteOrder order, size_t len);

DEFINE_ENCODE(int16_t)
DEFINE_ENCODE(int32_t)
DEFINE_ENCODE(int64_t)
DEFINE_ENCODE(float)
DEFINE_ENCODE(double)

#undef DEFINE_ENCODE

} // namespace torch::utils
