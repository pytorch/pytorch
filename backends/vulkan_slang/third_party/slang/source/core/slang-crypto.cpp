/*
 * MD5 implementation is based on:
 * http://openwall.info/wiki/people/solar/software/public-domain-source-code/md5
 * Original file header is at the bottom of this file.
 *
 * SHA1 implementation is based on:
 * https://github.com/983/SHA1
 * Original LICENSE is at the bottom of this file.
 */

#include "slang-crypto.h"

#include "../core/slang-char-util.h"

namespace Slang
{

// DigestUtil

/*static*/ String DigestUtil::digestToString(const void* digest, SlangInt digestSize)
{
    SLANG_ASSERT(digest && digestSize >= 0);

    static const char* hex = "0123456789abcdef";

    String str;
    const uint8_t* data = reinterpret_cast<const uint8_t*>(digest);
    for (SlangInt i = 0; i < digestSize; ++i)
    {
        str.append(hex[data[i] >> 4]);
        str.append(hex[data[i] & 0xf]);
    }
    return str;
}

/*static*/ bool DigestUtil::stringToDigest(
    const char* str,
    SlangInt strLength,
    void* digest,
    SlangInt digestSize)
{
    SLANG_ASSERT(str && strLength >= 0 && digest && digestSize >= 0);

    if (strLength != digestSize * 2)
    {
        ::memset(digest, 0, digestSize);
        return false;
    }

    uint8_t* data = reinterpret_cast<uint8_t*>(digest);
    for (SlangInt i = 0; i < digestSize; ++i)
    {
        int upper = CharUtil::getHexDigitValue(str[i * 2]);
        int lower = CharUtil::getHexDigitValue(str[i * 2 + 1]);
        if (upper == -1 || lower == -1)
        {
            ::memset(digest, 0, digestSize);
            return false;
        }
        data[i] = uint8_t(lower | upper << 4);
        ;
    }

    return true;
}

// MD5

MD5::MD5()
{
    init();
}

void MD5::init()
{
    m_lo = 0;
    m_hi = 0;
    m_a = 0x67452301;
    m_b = 0xefcdab89;
    m_c = 0x98badcfe;
    m_d = 0x10325476;
}

void MD5::update(const void* data, SlangSizeT size)
{
    uint32_t saved_lo;
    uint32_t used;
    uint32_t available;

    saved_lo = m_lo;
    if ((m_lo = (saved_lo + size) & 0x1fffffff) < saved_lo)
    {
        m_hi++;
    }
    m_hi += (uint32_t)size >> 29;

    used = saved_lo & 0x3f;

    if (used)
    {
        available = 64 - used;

        if (size < available)
        {
            ::memcpy(&m_buffer[used], data, size);
            return;
        }

        ::memcpy(&m_buffer[used], data, available);
        data = reinterpret_cast<const uint8_t*>(data) + available;
        size -= available;
        processBlock(m_buffer, 64);
    }

    if (size >= 64)
    {
        data = processBlock(data, size & ~(SlangInt)0x3f);
        size &= 0x3f;
    }

    ::memcpy(m_buffer, data, size);
}

MD5::Digest MD5::finalize()
{
    uint32_t used, available;

    used = m_lo & 0x3f;

    m_buffer[used++] = 0x80;

    available = 64 - used;

    if (available < 8)
    {
        ::memset(&m_buffer[used], 0, available);
        processBlock(m_buffer, 64);
        used = 0;
        available = 64;
    }

    ::memset(&m_buffer[used], 0, available - 8);

    m_lo <<= 3;

    m_buffer[56] = uint8_t(m_lo);
    m_buffer[57] = uint8_t(m_lo >> 8);
    m_buffer[58] = uint8_t(m_lo >> 16);
    m_buffer[59] = uint8_t(m_lo >> 24);
    m_buffer[60] = uint8_t(m_hi);
    m_buffer[61] = uint8_t(m_hi >> 8);
    m_buffer[62] = uint8_t(m_hi >> 16);
    m_buffer[63] = uint8_t(m_hi >> 24);

    processBlock(m_buffer, 64);

    Digest digest;
    digest.data[0] = m_a;
    digest.data[1] = m_b;
    digest.data[2] = m_c;
    digest.data[3] = m_d;

    return digest;
}

/*
 * The basic MD5 functions.
 *
 * F and G are optimized compared to their RFC 1321 definitions for
 * architectures that lack an AND-NOT instruction, just like in Colin Plumb's
 * implementation.
 */
#define F(x, y, z) ((z) ^ ((x) & ((y) ^ (z))))
#define G(x, y, z) ((y) ^ ((z) & ((x) ^ (y))))
#define H(x, y, z) (((x) ^ (y)) ^ (z))
#define H2(x, y, z) ((x) ^ ((y) ^ (z)))
#define I(x, y, z) ((y) ^ ((x) | ~(z)))

/*
 * The MD5 transformation for all four rounds.
 */
#define STEP(f, a, b, c, d, x, t, s)                           \
    (a) += f((b), (c), (d)) + (x) + (t);                       \
    (a) = (((a) << (s)) | (((a) & 0xffffffff) >> (32 - (s)))); \
    (a) += (b);

/*
 * SET reads 4 input bytes in little-endian byte order and stores them in a
 * properly aligned word in host byte order.
 */
#define SET(n)                                                                   \
    (m_block[(n)] = (uint32_t)ptr[(n) * 4] | ((uint32_t)ptr[(n) * 4 + 1] << 8) | \
                    ((uint32_t)ptr[(n) * 4 + 2] << 16) | ((uint32_t)ptr[(n) * 4 + 3] << 24))
#define GET(n) (m_block[(n)])

const void* MD5::processBlock(const void* data, SlangInt size)
{
    const unsigned char* ptr;
    ptr = (const unsigned char*)data;

    uint32_t a = m_a;
    uint32_t b = m_b;
    uint32_t c = m_c;
    uint32_t d = m_d;

    do
    {
        uint32_t saved_a = a;
        uint32_t saved_b = b;
        uint32_t saved_c = c;
        uint32_t saved_d = d;

        /* Round 1 */
        STEP(F, a, b, c, d, SET(0), 0xd76aa478, 7)
        STEP(F, d, a, b, c, SET(1), 0xe8c7b756, 12)
        STEP(F, c, d, a, b, SET(2), 0x242070db, 17)
        STEP(F, b, c, d, a, SET(3), 0xc1bdceee, 22)
        STEP(F, a, b, c, d, SET(4), 0xf57c0faf, 7)
        STEP(F, d, a, b, c, SET(5), 0x4787c62a, 12)
        STEP(F, c, d, a, b, SET(6), 0xa8304613, 17)
        STEP(F, b, c, d, a, SET(7), 0xfd469501, 22)
        STEP(F, a, b, c, d, SET(8), 0x698098d8, 7)
        STEP(F, d, a, b, c, SET(9), 0x8b44f7af, 12)
        STEP(F, c, d, a, b, SET(10), 0xffff5bb1, 17)
        STEP(F, b, c, d, a, SET(11), 0x895cd7be, 22)
        STEP(F, a, b, c, d, SET(12), 0x6b901122, 7)
        STEP(F, d, a, b, c, SET(13), 0xfd987193, 12)
        STEP(F, c, d, a, b, SET(14), 0xa679438e, 17)
        STEP(F, b, c, d, a, SET(15), 0x49b40821, 22)

        /* Round 2 */
        STEP(G, a, b, c, d, GET(1), 0xf61e2562, 5)
        STEP(G, d, a, b, c, GET(6), 0xc040b340, 9)
        STEP(G, c, d, a, b, GET(11), 0x265e5a51, 14)
        STEP(G, b, c, d, a, GET(0), 0xe9b6c7aa, 20)
        STEP(G, a, b, c, d, GET(5), 0xd62f105d, 5)
        STEP(G, d, a, b, c, GET(10), 0x02441453, 9)
        STEP(G, c, d, a, b, GET(15), 0xd8a1e681, 14)
        STEP(G, b, c, d, a, GET(4), 0xe7d3fbc8, 20)
        STEP(G, a, b, c, d, GET(9), 0x21e1cde6, 5)
        STEP(G, d, a, b, c, GET(14), 0xc33707d6, 9)
        STEP(G, c, d, a, b, GET(3), 0xf4d50d87, 14)
        STEP(G, b, c, d, a, GET(8), 0x455a14ed, 20)
        STEP(G, a, b, c, d, GET(13), 0xa9e3e905, 5)
        STEP(G, d, a, b, c, GET(2), 0xfcefa3f8, 9)
        STEP(G, c, d, a, b, GET(7), 0x676f02d9, 14)
        STEP(G, b, c, d, a, GET(12), 0x8d2a4c8a, 20)

        /* Round 3 */
        STEP(H, a, b, c, d, GET(5), 0xfffa3942, 4)
        STEP(H2, d, a, b, c, GET(8), 0x8771f681, 11)
        STEP(H, c, d, a, b, GET(11), 0x6d9d6122, 16)
        STEP(H2, b, c, d, a, GET(14), 0xfde5380c, 23)
        STEP(H, a, b, c, d, GET(1), 0xa4beea44, 4)
        STEP(H2, d, a, b, c, GET(4), 0x4bdecfa9, 11)
        STEP(H, c, d, a, b, GET(7), 0xf6bb4b60, 16)
        STEP(H2, b, c, d, a, GET(10), 0xbebfbc70, 23)
        STEP(H, a, b, c, d, GET(13), 0x289b7ec6, 4)
        STEP(H2, d, a, b, c, GET(0), 0xeaa127fa, 11)
        STEP(H, c, d, a, b, GET(3), 0xd4ef3085, 16)
        STEP(H2, b, c, d, a, GET(6), 0x04881d05, 23)
        STEP(H, a, b, c, d, GET(9), 0xd9d4d039, 4)
        STEP(H2, d, a, b, c, GET(12), 0xe6db99e5, 11)
        STEP(H, c, d, a, b, GET(15), 0x1fa27cf8, 16)
        STEP(H2, b, c, d, a, GET(2), 0xc4ac5665, 23)

        /* Round 4 */
        STEP(I, a, b, c, d, GET(0), 0xf4292244, 6)
        STEP(I, d, a, b, c, GET(7), 0x432aff97, 10)
        STEP(I, c, d, a, b, GET(14), 0xab9423a7, 15)
        STEP(I, b, c, d, a, GET(5), 0xfc93a039, 21)
        STEP(I, a, b, c, d, GET(12), 0x655b59c3, 6)
        STEP(I, d, a, b, c, GET(3), 0x8f0ccc92, 10)
        STEP(I, c, d, a, b, GET(10), 0xffeff47d, 15)
        STEP(I, b, c, d, a, GET(1), 0x85845dd1, 21)
        STEP(I, a, b, c, d, GET(8), 0x6fa87e4f, 6)
        STEP(I, d, a, b, c, GET(15), 0xfe2ce6e0, 10)
        STEP(I, c, d, a, b, GET(6), 0xa3014314, 15)
        STEP(I, b, c, d, a, GET(13), 0x4e0811a1, 21)
        STEP(I, a, b, c, d, GET(4), 0xf7537e82, 6)
        STEP(I, d, a, b, c, GET(11), 0xbd3af235, 10)
        STEP(I, c, d, a, b, GET(2), 0x2ad7d2bb, 15)
        STEP(I, b, c, d, a, GET(9), 0xeb86d391, 21)

        a += saved_a;
        b += saved_b;
        c += saved_c;
        d += saved_d;

        ptr += 64;
    } while (size -= 64);

    m_a = a;
    m_b = b;
    m_c = c;
    m_d = d;

    return ptr;
}

#undef F
#undef G
#undef H
#undef H2
#undef I
#undef STEP
#undef SET
#undef GET

/*static*/ MD5::Digest MD5::compute(const void* data, SlangInt size)
{
    MD5 md5;
    md5.update(data, size);
    return md5.finalize();
}

// SHA1

SHA1::SHA1()
{
    init();
}

void SHA1::init()
{
    m_index = 0;
    m_bits = 0;
    m_state[0] = 0x67452301;
    m_state[1] = 0xefcdab89;
    m_state[2] = 0x98badcfe;
    m_state[3] = 0x10325476;
    m_state[4] = 0xc3d2e1f0;
}

void SHA1::update(const void* data, SlangSizeT len)
{
    if (!data || len <= 0)
    {
        return;
    }

    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(data);

    // Fill up buffer if not full.
    while (len > 0 && m_index != 0)
    {
        addByte(*ptr++);
        m_bits += 8;
        len--;
    }

    // Process full blocks.
    while (len >= sizeof(m_buf))
    {
        processBlock(ptr);
        ptr += sizeof(m_buf);
        len -= sizeof(m_buf);
        m_bits += sizeof(m_buf) * 8;
    }

    // Process remaining bytes.
    while (len > 0)
    {
        addByte(*ptr++);
        m_bits += 8;
        len--;
    }
}

SHA1::Digest SHA1::finalize()
{
    // Finalize with 0x80, some zero padding and the length in bits.
    addByte(0x80);
    while (m_index % 64 != 56)
    {
        addByte(0);
    }
    for (int i = 7; i >= 0; --i)
    {
        addByte(uint8_t(m_bits >> i * 8));
    }

    Digest digest;
    uint8_t* data = reinterpret_cast<uint8_t*>(digest.data);
    for (int i = 0; i < 5; i++)
    {
        for (int j = 3; j >= 0; j--)
        {
            data[i * 4 + j] = (m_state[i] >> ((3 - j) * 8)) & 0xff;
        }
    }

    return digest;
}

void SHA1::addByte(uint8_t byte)
{
    m_buf[m_index++] = byte;

    if (m_index >= sizeof(m_buf))
    {
        m_index = 0;
        processBlock(m_buf);
    }
}

void SHA1::processBlock(const uint8_t* ptr)
{
    auto rol32 = [](uint32_t x, uint32_t n) { return (x << n) | (x >> (32 - n)); };

    auto makeWord = [](const uint8_t* p)
    {
        return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) |
               (uint32_t)p[3];
    };

    const uint32_t c0 = 0x5a827999;
    const uint32_t c1 = 0x6ed9eba1;
    const uint32_t c2 = 0x8f1bbcdc;
    const uint32_t c3 = 0xca62c1d6;

    uint32_t a = m_state[0];
    uint32_t b = m_state[1];
    uint32_t c = m_state[2];
    uint32_t d = m_state[3];
    uint32_t e = m_state[4];

    uint32_t w[16];

    for (size_t i = 0; i < 16; i++)
    {
        w[i] = makeWord(ptr + i * 4);
    }

#define SHA1_LOAD(i) \
    w[i & 15] = rol32(w[(i + 13) & 15] ^ w[(i + 8) & 15] ^ w[(i + 2) & 15] ^ w[i & 15], 1);
#define SHA1_ROUND_0(v, u, x, y, z, i)                       \
    z += ((u & (x ^ y)) ^ y) + w[i & 15] + c0 + rol32(v, 5); \
    u = rol32(u, 30);
#define SHA1_ROUND_1(v, u, x, y, z, i)                                    \
    SHA1_LOAD(i) z += ((u & (x ^ y)) ^ y) + w[i & 15] + c0 + rol32(v, 5); \
    u = rol32(u, 30);
#define SHA1_ROUND_2(v, u, x, y, z, i)                            \
    SHA1_LOAD(i) z += (u ^ x ^ y) + w[i & 15] + c1 + rol32(v, 5); \
    u = rol32(u, 30);
#define SHA1_ROUND_3(v, u, x, y, z, i)                                          \
    SHA1_LOAD(i) z += (((u | x) & y) | (u & x)) + w[i & 15] + c2 + rol32(v, 5); \
    u = rol32(u, 30);
#define SHA1_ROUND_4(v, u, x, y, z, i)                            \
    SHA1_LOAD(i) z += (u ^ x ^ y) + w[i & 15] + c3 + rol32(v, 5); \
    u = rol32(u, 30);

    SHA1_ROUND_0(a, b, c, d, e, 0);
    SHA1_ROUND_0(e, a, b, c, d, 1);
    SHA1_ROUND_0(d, e, a, b, c, 2);
    SHA1_ROUND_0(c, d, e, a, b, 3);
    SHA1_ROUND_0(b, c, d, e, a, 4);
    SHA1_ROUND_0(a, b, c, d, e, 5);
    SHA1_ROUND_0(e, a, b, c, d, 6);
    SHA1_ROUND_0(d, e, a, b, c, 7);
    SHA1_ROUND_0(c, d, e, a, b, 8);
    SHA1_ROUND_0(b, c, d, e, a, 9);
    SHA1_ROUND_0(a, b, c, d, e, 10);
    SHA1_ROUND_0(e, a, b, c, d, 11);
    SHA1_ROUND_0(d, e, a, b, c, 12);
    SHA1_ROUND_0(c, d, e, a, b, 13);
    SHA1_ROUND_0(b, c, d, e, a, 14);
    SHA1_ROUND_0(a, b, c, d, e, 15);
    SHA1_ROUND_1(e, a, b, c, d, 16);
    SHA1_ROUND_1(d, e, a, b, c, 17);
    SHA1_ROUND_1(c, d, e, a, b, 18);
    SHA1_ROUND_1(b, c, d, e, a, 19);
    SHA1_ROUND_2(a, b, c, d, e, 20);
    SHA1_ROUND_2(e, a, b, c, d, 21);
    SHA1_ROUND_2(d, e, a, b, c, 22);
    SHA1_ROUND_2(c, d, e, a, b, 23);
    SHA1_ROUND_2(b, c, d, e, a, 24);
    SHA1_ROUND_2(a, b, c, d, e, 25);
    SHA1_ROUND_2(e, a, b, c, d, 26);
    SHA1_ROUND_2(d, e, a, b, c, 27);
    SHA1_ROUND_2(c, d, e, a, b, 28);
    SHA1_ROUND_2(b, c, d, e, a, 29);
    SHA1_ROUND_2(a, b, c, d, e, 30);
    SHA1_ROUND_2(e, a, b, c, d, 31);
    SHA1_ROUND_2(d, e, a, b, c, 32);
    SHA1_ROUND_2(c, d, e, a, b, 33);
    SHA1_ROUND_2(b, c, d, e, a, 34);
    SHA1_ROUND_2(a, b, c, d, e, 35);
    SHA1_ROUND_2(e, a, b, c, d, 36);
    SHA1_ROUND_2(d, e, a, b, c, 37);
    SHA1_ROUND_2(c, d, e, a, b, 38);
    SHA1_ROUND_2(b, c, d, e, a, 39);
    SHA1_ROUND_3(a, b, c, d, e, 40);
    SHA1_ROUND_3(e, a, b, c, d, 41);
    SHA1_ROUND_3(d, e, a, b, c, 42);
    SHA1_ROUND_3(c, d, e, a, b, 43);
    SHA1_ROUND_3(b, c, d, e, a, 44);
    SHA1_ROUND_3(a, b, c, d, e, 45);
    SHA1_ROUND_3(e, a, b, c, d, 46);
    SHA1_ROUND_3(d, e, a, b, c, 47);
    SHA1_ROUND_3(c, d, e, a, b, 48);
    SHA1_ROUND_3(b, c, d, e, a, 49);
    SHA1_ROUND_3(a, b, c, d, e, 50);
    SHA1_ROUND_3(e, a, b, c, d, 51);
    SHA1_ROUND_3(d, e, a, b, c, 52);
    SHA1_ROUND_3(c, d, e, a, b, 53);
    SHA1_ROUND_3(b, c, d, e, a, 54);
    SHA1_ROUND_3(a, b, c, d, e, 55);
    SHA1_ROUND_3(e, a, b, c, d, 56);
    SHA1_ROUND_3(d, e, a, b, c, 57);
    SHA1_ROUND_3(c, d, e, a, b, 58);
    SHA1_ROUND_3(b, c, d, e, a, 59);
    SHA1_ROUND_4(a, b, c, d, e, 60);
    SHA1_ROUND_4(e, a, b, c, d, 61);
    SHA1_ROUND_4(d, e, a, b, c, 62);
    SHA1_ROUND_4(c, d, e, a, b, 63);
    SHA1_ROUND_4(b, c, d, e, a, 64);
    SHA1_ROUND_4(a, b, c, d, e, 65);
    SHA1_ROUND_4(e, a, b, c, d, 66);
    SHA1_ROUND_4(d, e, a, b, c, 67);
    SHA1_ROUND_4(c, d, e, a, b, 68);
    SHA1_ROUND_4(b, c, d, e, a, 69);
    SHA1_ROUND_4(a, b, c, d, e, 70);
    SHA1_ROUND_4(e, a, b, c, d, 71);
    SHA1_ROUND_4(d, e, a, b, c, 72);
    SHA1_ROUND_4(c, d, e, a, b, 73);
    SHA1_ROUND_4(b, c, d, e, a, 74);
    SHA1_ROUND_4(a, b, c, d, e, 75);
    SHA1_ROUND_4(e, a, b, c, d, 76);
    SHA1_ROUND_4(d, e, a, b, c, 77);
    SHA1_ROUND_4(c, d, e, a, b, 78);
    SHA1_ROUND_4(b, c, d, e, a, 79);

#undef SHA1_LOAD
#undef SHA1_ROUND_0
#undef SHA1_ROUND_1
#undef SHA1_ROUND_2
#undef SHA1_ROUND_3
#undef SHA1_ROUND_4

    m_state[0] += a;
    m_state[1] += b;
    m_state[2] += c;
    m_state[3] += d;
    m_state[4] += e;
}

/* static */ SHA1::Digest SHA1::compute(const void* data, SlangInt size)
{
    SHA1 sha1;
    sha1.update(data, size);
    return sha1.finalize();
}

} // namespace Slang


/*
 * This is an OpenSSL-compatible implementation of the RSA Data Security, Inc.
 * MD5 Message-Digest Algorithm (RFC 1321).
 *
 * Homepage:
 * http://openwall.info/wiki/people/solar/software/public-domain-source-code/md5
 *
 * Author:
 * Alexander Peslyak, better known as Solar Designer <solar at openwall.com>
 *
 * This software was written by Alexander Peslyak in 2001.  No copyright is
 * claimed, and the software is hereby placed in the public domain.
 * In case this attempt to disclaim copyright and place the software in the
 * public domain is deemed null and void, then the software is
 * Copyright (c) 2001 Alexander Peslyak and it is hereby released to the
 * general public under the following terms:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 *
 * There's ABSOLUTELY NO WARRANTY, express or implied.
 *
 * (This is a heavily cut-down "BSD license".)
 *
 * This differs from Colin Plumb's older public domain implementation in that
 * no exactly 32-bit integer data type is required (any 32-bit or wider
 * unsigned integer data type will do), there's no compile-time endianness
 * configuration, and the function prototypes match OpenSSL's.  No code from
 * Colin Plumb's implementation has been reused; this comment merely compares
 * the properties of the two independent implementations.
 *
 * The primary goals of this implementation are portability and ease of use.
 * It is meant to be fast, but not as fast as possible.  Some known
 * optimizations are not included to reduce source code size and avoid
 * compile-time configuration.
 */

/*
 * This is free and unencumbered software released into the public domain.
 *
 * Anyone is free to copy, modify, publish, use, compile, sell, or
 * distribute this software, either in source code form or as a compiled
 * binary, for any purpose, commercial or non-commercial, and by any
 * means.
 *
 * In jurisdictions that recognize copyright laws, the author or authors
 * of this software dedicate any and all copyright interest in the
 * software to the public domain. We make this dedication for the benefit
 * of the public at large and to the detriment of our heirs and
 * successors. We intend this dedication to be an overt act of
 * relinquishment in perpetuity of all present and future rights to this
 * software under copyright law.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * For more information, please refer to <http://unlicense.org>
 */
