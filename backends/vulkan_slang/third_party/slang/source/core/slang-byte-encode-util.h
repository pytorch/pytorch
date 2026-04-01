#ifndef SLANG_CORE_BYTE_ENCODE_UTIL_H
#define SLANG_CORE_BYTE_ENCODE_UTIL_H

#include "slang-list.h"

namespace Slang
{

struct ByteEncodeUtil
{
    enum
    {
        kMaxLiteEncodeUInt16 = 3, /// One byte for prefix, the remaining 2 bytes hold the value
        kMaxLiteEncodeUInt32 = 5, /// One byte for prefix, the remaining 4 bytes hold the value
        // Cut values for 'Lite' encoding style
        kLiteCut1 = 185,
        kLiteCut2 = 249,
    };

    /** Find the most significant bit for 8 bits
    @param v The value to find most significant bit on
    @return The most significant bit, or -1 if no bits are set
    */
    SLANG_FORCE_INLINE static int calcMsb8(uint32_t v);

    /** Find the most significant bit for 32 bits
    @param v The value to find most significant bit on
    @return The most significant bit, or -1 if no bits are set
    */
    SLANG_FORCE_INLINE static int calcMsb32(uint32_t v);

    /** Calculates the 'most significant' byte ie the highest bytes that is non zero.
     Note return value is *undefined* if in is 0.
     @param in Value - cannot be 0.
     @return The byte index of the highest byte that is non zero.
     */
    SLANG_FORCE_INLINE static int calcNonZeroMsByte32(uint32_t in);

    /** Calculates the 'most significant' byte ie the highest bytes that is non zero.
    @param in Value - cannot be 0.
    @return The byte index of the highest byte that is non zero.
    */
    SLANG_FORCE_INLINE static int calcMsByte32(uint32_t in);

    /// Calculate the size of encoding bytes
    static size_t calcEncodeLiteSizeUInt32(const uint32_t* in, size_t num);

    /// Calculate the size of a single value
    static size_t calcEncodeLiteSizeUInt32(uint32_t in);

    /** Encodes a uint32_t as an integer
     @return the number of bytes needed to encode */
    static int encodeLiteUInt32(uint32_t in, uint8_t out[kMaxLiteEncodeUInt32]);

    /** Decode a lite encoding.
     @param in The lite encoded bytes
     @param out Value constructed
    @return number of bytes on in consumed */
    static int decodeLiteUInt32(const uint8_t* in, uint32_t* out);

    /** Encode an array of uint32_t
    @param in The values to encode
    @param num The amount of values to encode
    @param encodeOut The buffer to hold the encoded value. MUST be large enough to hold the encoding
    @return The size of the encoding in bytes
    */
    static size_t encodeLiteUInt32(const uint32_t* in, size_t num, uint8_t* encodeOut);

    /** Encode an array of uint32_t
    @param in The values to encode
    @param num The amount of values to encode
    @param encodeOut The buffer to hold the encoded value.
    */
    static void encodeLiteUInt32(const uint32_t* in, size_t num, List<uint8_t>& encodeOut);

    /** Encode an array of uint32_t
    @param encodeIn The encoded values
    @param numValues The amount of values to be decoded (NOTE! This is the number of valuesOut, not
    encodeIn)
    @param valuesOut The buffer to hold the encoded value. MUST be large enough to hold the encoding
    @return The amount of bytes decoded
    */
    static size_t decodeLiteUInt32(const uint8_t* encodeIn, size_t numValues, uint32_t* valuesOut);

    /// Table that maps 8 bits to it's most significant bit. If 0 returns -1.
    static const int8_t s_msb8[256];
};

#if SLANG_VC
// Works on ARM and x86/64 on visual studio compiler

// ---------------------------------------------------------------------------
SLANG_FORCE_INLINE int ByteEncodeUtil::calcNonZeroMsByte32(uint32_t in)
{
    SLANG_ASSERT(in != 0);
    // Can use intrinsic
    // https://msdn.microsoft.com/en-us/library/fbxyd7zd.aspx
    unsigned long index;
    _BitScanReverse(&index, in);
    return index >> 3;
}

// ---------------------------------------------------------------------------
SLANG_FORCE_INLINE int ByteEncodeUtil::calcMsByte32(uint32_t in)
{
    if (in == 0)
    {
        return -1;
    }
    // Can use intrinsic
    // https://msdn.microsoft.com/en-us/library/fbxyd7zd.aspx
    unsigned long index;
    _BitScanReverse(&index, in);
    return index >> 3;
}

// ---------------------------------------------------------------------------
SLANG_FORCE_INLINE /* static */ int ByteEncodeUtil::calcMsb8(uint32_t v)
{
    SLANG_ASSERT((v & 0xffffff00) == 0);
    if (v == 0)
    {
        return -1;
    }
    unsigned long index;
    _BitScanReverse(&index, v);
    return index;
}

// ---------------------------------------------------------------------------
SLANG_FORCE_INLINE /* static */ int ByteEncodeUtil::calcMsb32(uint32_t v)
{
    if (v == 0)
    {
        return -1;
    }
    unsigned long index;
    _BitScanReverse(&index, v);
    return index;
}

#else

// ---------------------------------------------------------------------------
SLANG_FORCE_INLINE /* static */ int ByteEncodeUtil::calcNonZeroMsByte32(uint32_t in)
{
    return (in & 0xffff0000) ? ((in & 0xff000000) ? 3 : 2) : ((in & 0x0000ff00) ? 1 : 0);
}

// ---------------------------------------------------------------------------
SLANG_FORCE_INLINE /* static */ int ByteEncodeUtil::calcMsByte32(uint32_t in)
{
    return (in & 0xffff0000) ? ((in & 0xff000000) ? 3 : 2)
                             : ((in & 0x0000ff00) ? 1 : ((in == 0) ? -1 : 0));
}

// ---------------------------------------------------------------------------
SLANG_FORCE_INLINE /* static */ int ByteEncodeUtil::calcMsb8(uint32_t v)
{
    SLANG_ASSERT((v & 0xffffff00) == 0);
    return s_msb8[v];
}

// ---------------------------------------------------------------------------
SLANG_FORCE_INLINE /* static */ int ByteEncodeUtil::calcMsb32(uint32_t v)
{
    return (v & 0xffff0000) ? ((v & 0xff000000) ? s_msb8[v >> 24] + 24 : s_msb8[v >> 16] + 16)
                            : ((v & 0x0000ff00) ? s_msb8[v >> 8] + 8 : s_msb8[v]);
}

#endif

// ---------------------------------------------------------------------------
inline /* static */ size_t ByteEncodeUtil::calcEncodeLiteSizeUInt32(uint32_t v)
{
    if (v < kLiteCut1)
    {
        return 1;
    }
    else if (v <= kLiteCut1 + 255 * (kLiteCut2 - 1 - kLiteCut1))
    {
        return 2;
    }
    else
    {
        return calcNonZeroMsByte32(v) + 2;
    }
}

} // namespace Slang

#endif // SLANG_BYTE_ENCODE_UTIL_H
