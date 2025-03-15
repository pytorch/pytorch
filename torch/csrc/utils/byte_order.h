#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Half.h>
#include <torch/csrc/Export.h>
#include <cstddef>
#include <cstdint>

#ifdef __FreeBSD__
#include <sys/endian.h>
#include <sys/types.h>
#define thp_bswap16(x) bswap16(x)
#define thp_bswap32(x) bswap32(x)
#define thp_bswap64(x) bswap64(x)
#elif defined(__APPLE__)
#include <libkern/OSByteOrder.h>
#define thp_bswap16(x) OSSwapInt16(x)
#define thp_bswap32(x) OSSwapInt32(x)
#define thp_bswap64(x) OSSwapInt64(x)
#elif defined(__GNUC__) && !defined(__MINGW32__) && !defined(_AIX)
#include <byteswap.h>
#define thp_bswap16(x) bswap_16(x)
#define thp_bswap32(x) bswap_32(x)
#define thp_bswap64(x) bswap_64(x)
#elif defined _WIN32 || defined _WIN64
#define thp_bswap16(x) _byteswap_ushort(x)
#define thp_bswap32(x) _byteswap_ulong(x)
#define thp_bswap64(x) _byteswap_uint64(x)
#endif

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define to_be16(x) thp_bswap16(x)
#define from_be16(x) thp_bswap16(x)
#define to_be32(x) thp_bswap32(x)
#define from_be32(x) thp_bswap32(x)
#define to_be64(x) thp_bswap64(x)
#define from_be64(x) thp_bswap64(x)
#define to_le16(x) (x)
#define from_le16(x) (x)
#define to_le32(x) (x)
#define from_le32(x) (x)
#define to_le64(x) (x)
#define from_le64(x) (x)
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define to_be16(x) (x)
#define from_be16(x) (x)
#define to_be32(x) (x)
#define from_be32(x) (x)
#define to_be64(x) (x)
#define from_be64(x) (x)
#define to_le16(x) thp_bswap16(x)
#define from_le16(x) thp_bswap16(x)
#define to_le32(x) thp_bswap32(x)
#define from_le32(x) thp_bswap32(x)
#define to_le64(x) thp_bswap64(x)
#define from_le64(x) thp_bswap64(x)
#else
#error Unexpected or undefined __BYTE_ORDER__
#endif

#ifdef _AIX
#include <stdint.h>

// Macro to swap endianness for 16-bit values
#define SWAP_ENDIAN_16(val) (((val) >> 8) & 0x00FF) | (((val) << 8) & 0xFF00)

// Macro to swap endianness for 32-bit values
#define SWAP_ENDIAN_32(val)\
((((val) >> 24) & 0x000000FF) | (((val) >> 8) & 0x0000FF00) | \
(((val) << 8) & 0x00FF0000) | (((val) << 24) & 0xFF000000))

// Macro to swap endianness for 64-bit values
#define SWAP_ENDIAN_64(val)\
((((val) >> 56) & 0x00000000000000FFULL) | \
(((val) >> 40) & 0x000000000000FF00ULL) | \
(((val) >> 24) & 0x0000000000FF0000ULL) | \
(((val) >> 8)& 0x00000000FF000000ULL) |\
(((val) << 8)& 0x000000FF00000000ULL) |  \
(((val) << 24) & 0x0000FF0000000000ULL) | \
(((val) << 40) & 0x00FF000000000000ULL) | \
(((val) << 56) & 0xFF00000000000000ULL))

#define thp_bswap16(x) SWAP_ENDIAN_16(x)
#define thp_bswap32(x) SWAP_ENDIAN_32(x)
#define thp_bswap64(x) SWAP_ENDIAN_64(x)
#endif // _AIX

namespace torch::utils {

enum THPByteOrder { THP_LITTLE_ENDIAN = 0, THP_BIG_ENDIAN = 1 };

TORCH_API THPByteOrder THP_nativeByteOrder();

template <typename T, typename U>
TORCH_API void THP_decodeBuffer(T* dst, const uint8_t* src, U type, size_t len);

template <typename T>
TORCH_API void THP_encodeBuffer(
    uint8_t* dst,
    const T* src,
    THPByteOrder order,
    size_t len);

} // namespace torch::utils
