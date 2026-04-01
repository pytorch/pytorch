#ifndef SLANG_CORE_MATH_H
#define SLANG_CORE_MATH_H

#include "slang.h"

#include <cmath>

namespace Slang
{
// Some handy constants

// The largest positive (or negative) number
#define SLANG_HALF_MAX 65504.0f
// Smallest (denormalized) value. 1 / 2^24
#define SLANG_HALF_SUB_NORMAL_MIN (1.0f / 16777216.0f)

class Math
{
public:
    // Use to fix type punning issues with strict aliasing
    union FloatIntUnion
    {
        float fvalue;
        int ivalue;

        SLANG_FORCE_INLINE static FloatIntUnion makeFromInt(int i)
        {
            FloatIntUnion cast;
            cast.ivalue = i;
            return cast;
        }
        SLANG_FORCE_INLINE static FloatIntUnion makeFromFloat(float f)
        {
            FloatIntUnion cast;
            cast.fvalue = f;
            return cast;
        }
    };
    union DoubleInt64Union
    {
        double dvalue;
        int64_t ivalue;
        SLANG_FORCE_INLINE static DoubleInt64Union makeFromInt64(int64_t i)
        {
            DoubleInt64Union cast;
            cast.ivalue = i;
            return cast;
        }
        SLANG_FORCE_INLINE static DoubleInt64Union makeFromDouble(double d)
        {
            DoubleInt64Union cast;
            cast.dvalue = d;
            return cast;
        }
    };

    static const float Pi;

    template<typename T>
    static T Abs(T a)
    {
        return (a < 0) ? -a : a;
    }

    template<typename T>
    static T Min(const T& v1, const T& v2)
    {
        return v1 < v2 ? v1 : v2;
    }
    template<typename T>
    static T Max(const T& v1, const T& v2)
    {
        return v1 > v2 ? v1 : v2;
    }
    template<typename T>
    static T Min(const T& v1, const T& v2, const T& v3)
    {
        return Min(v1, Min(v2, v3));
    }
    template<typename T>
    static T Max(const T& v1, const T& v2, const T& v3)
    {
        return Max(v1, Max(v2, v3));
    }
    template<typename T>
    static T Clamp(const T& val, const T& vmin, const T& vmax)
    {
        if (val < vmin)
            return vmin;
        else if (val > vmax)
            return vmax;
        else
            return val;
    }

    static inline int FastFloor(float x)
    {
        int i = (int)x;
        return i - (i > x);
    }

    static inline int FastFloor(double x)
    {
        int i = (int)x;
        return i - (i > x);
    }

    static inline int IsNaN(float x) { return std::isnan(x); }

    static inline int IsInf(float x) { return std::isinf(x); }

    static inline unsigned int Ones32(unsigned int x)
    {
        /* 32-bit recursive reduction using SWAR...
            but first step is mapping 2-bit values
            into sum of 2 1-bit values in sneaky way
        */
        x -= ((x >> 1) & 0x55555555);
        x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
        x = (((x >> 4) + x) & 0x0f0f0f0f);
        x += (x >> 8);
        x += (x >> 16);
        return (x & 0x0000003f);
    }

    static inline unsigned int Log2Floor(unsigned int x)
    {
        x |= (x >> 1);
        x |= (x >> 2);
        x |= (x >> 4);
        x |= (x >> 8);
        x |= (x >> 16);
        return (Ones32(x >> 1));
    }

    static inline unsigned int Log2Ceil(unsigned int x)
    {
        int y = (x & (x - 1));
        y |= -y;
        y >>= (32 - 1);
        x |= (x >> 1);
        x |= (x >> 2);
        x |= (x >> 4);
        x |= (x >> 8);
        x |= (x >> 16);
        return (Ones32(x >> 1) - y);
    }
    /*
    static inline int Log2(float x)
    {
        unsigned int ix = (unsigned int&)x;
        unsigned int exp = (ix >> 23) & 0xFF;
        int log2 = (unsigned int)(exp) - 127;

        return log2;
    }
    */

    static bool AreNearlyEqual(double a, double b, double epsilon)
    {
        // If they are equal then we are done
        if (a == b)
        {
            return true;
        }

        const double absA = Abs(a);
        const double absB = Abs(b);
        const double diff = Abs(a - b);

        // https://en.wikipedia.org/wiki/Double_precision_floating-point_format
        const double minNormal = 2.2250738585072014e-308;
        // Either a or b are very close to being zero, so doing relative comparison isn't really
        // appropriate
        if (a == 0.0 || b == 0.0 || (absA + absB < minNormal))
        {
            return diff < (epsilon * minNormal);
        }
        else
        {
            // Calculate a relative relative error
            return diff < epsilon * (absA + absB);
        }
    }

    template<typename T>
    static T getLowestBit(T val)
    {
        return val & (-val);
    }
};
inline int FloatAsInt(float val)
{
    return Math::FloatIntUnion::makeFromFloat(val).ivalue;
}
inline float IntAsFloat(int val)
{
    return Math::FloatIntUnion::makeFromInt(val).fvalue;
}

SLANG_FORCE_INLINE int64_t DoubleAsInt64(double val)
{
    return Math::DoubleInt64Union::makeFromDouble(val).ivalue;
}
SLANG_FORCE_INLINE double Int64AsDouble(int64_t value)
{
    return Math::DoubleInt64Union::makeFromInt64(value).dvalue;
}

inline unsigned short FloatToHalf(float val)
{
    const auto x = FloatAsInt(val);

    unsigned short bits = (x >> 16) & 0x8000;
    unsigned short m = (x >> 12) & 0x07ff;
    unsigned int e = (x >> 23) & 0xff;
    if (e < 103)
        return bits;
    if (e > 142)
    {
        bits |= 0x7c00u;
        bits |= e == 255 && (x & 0x007fffffu);
        return bits;
    }
    if (e < 113)
    {
        m |= 0x0800u;
        bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
        return bits;
    }
    bits |= ((e - 112) << 10) | (m >> 1);
    bits += m & 1;
    return bits;
}

inline float HalfToFloat(unsigned short input)
{
    static const auto magic = Math::FloatIntUnion::makeFromInt((127 + (127 - 15)) << 23);
    static const auto was_infnan = Math::FloatIntUnion::makeFromInt((127 + 16) << 23);
    Math::FloatIntUnion o;
    o.ivalue = (input & 0x7fff) << 13; // exponent/mantissa bits
    o.fvalue *= magic.fvalue;          // exponent adjust
    if (o.fvalue >= was_infnan.fvalue) // make sure Inf/NaN survive
        o.ivalue |= 255 << 23;
    o.ivalue |= (input & 0x8000) << 16; // sign bit
    return o.fvalue;
}

class Random
{
private:
    unsigned int seed;

public:
    Random(int seed) { this->seed = seed; }
    int Next() // random between 0 and RandMax (currently 0x7fff)
    {
        return ((seed = ((seed << 12) + 150889L) % 714025) & 0x7fff);
    }
    int Next(int min, int max) // inclusive min, exclusive max
    {
        unsigned int a = ((seed = ((seed << 12) + 150889L) % 714025) & 0xFFFF);
        unsigned int b = ((seed = ((seed << 12) + 150889L) % 714025) & 0xFFFF);
        unsigned int r = (a << 16) + b;
        return min + r % (max - min);
    }
    float NextFloat() { return ((Next() << 15) + Next()) / ((float)(1 << 30)); }
    float NextFloat(float valMin, float valMax) { return valMin + (valMax - valMin) * NextFloat(); }
    static int RandMax() { return 0x7fff; }
};
} // namespace Slang

#endif
