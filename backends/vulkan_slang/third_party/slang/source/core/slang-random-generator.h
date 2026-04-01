#ifndef SLANG_CORE_RANDOM_GENERATOR_H
#define SLANG_CORE_RANDOM_GENERATOR_H

#include "slang-smart-pointer.h"
#include "slang.h"

#include <stdlib.h>
#include <string.h>

namespace Slang
{

class RandomGenerator : public RefObject
{
public:
    /// Make a copy of the generator in the same state
    virtual RandomGenerator* clone() = 0;

    /// Reset with a seed
    virtual void reset(int32_t seed) = 0;
    /// Next int32_t random number
    virtual int32_t nextInt32() = 0;
    /// Next int64_t random number
    virtual int64_t nextInt64();

    /// Get a 0-1 range floating point
    virtual float nextUnitFloat32();

    /// Get the next bool
    virtual bool nextBool();

    /// Get multiple int32s
    virtual void nextInt32s(int32_t* dst, size_t count) = 0;

    /// Next uint32_t
    uint32_t nextUInt32() { return uint32_t(nextInt32()); }

    /// Next Int32 which can only be positive
    int32_t nextPositiveInt32() { return nextInt32() & 0x7fffffff; }
    /// Next Int64 which can only be positive
    int64_t nextPositiveInt64() { return nextInt64() & SLANG_INT64(0x7fffffffffffffff); }

    /// Returns value up to BUT NOT INCLUDING maxValue.
    int32_t nextInt32UpTo(int32_t maxValue)
    {
        assert(maxValue > 0);
        return (maxValue <= 1) ? 0 : (nextPositiveInt32() % maxValue);
    }

    /// Returns value from min up to BUT NOT INCLUDING max.
    int32_t nextInt32InRange(int32_t min, int32_t max);

    /// Returns value from min up to BUT NOT INCLUDING max
    uint32_t nextUInt32InRange(uint32_t min, uint32_t max);

    /// Returns value up to BUT NOT INCLUDING maxValue
    int64_t nextInt64UpTo(int64_t maxValue)
    {
        assert(maxValue > 0);
        return (maxValue <= 1) ? 0 : (nextPositiveInt64() % maxValue);
    }

    /// Returns value from min up to BUT NOT INCLUDING max
    int64_t nextInt64InRange(int64_t min, int64_t max);

    /// Fill with random data.
    /// NOTE! Output is only identical bytes if generator in same state *and* size_t(dst) & 3 is the
    /// same on calls.
    void nextData(void* dst, size_t size);

    /// Create a RandomGenerator with specified seed using default generator type
    static RandomGenerator* create(int32_t seed);
};

/* Mersenne Twister random number generator
https://en.wikipedia.org/wiki/Mersenne_Twister
*/
class Mt19937RandomGenerator : public RandomGenerator
{
public:
    typedef Mt19937RandomGenerator ThisType;

    enum
    {
        kNumEntries = 624
    };

    Mt19937RandomGenerator* clone() SLANG_OVERRIDE { return new ThisType(*this); }
    void reset(int32_t seed) SLANG_OVERRIDE;
    int32_t nextInt32() SLANG_OVERRIDE;
    void nextInt32s(int32_t* dst, size_t count) SLANG_OVERRIDE;

    /// Ctor
    Mt19937RandomGenerator();
    Mt19937RandomGenerator(const ThisType& rhs);
    explicit Mt19937RandomGenerator(int32_t seed);

    /// Assignment
    void operator=(const ThisType& rhs)
    {
        m_index = rhs.m_index;
        ::memcpy(m_mt, rhs.m_mt, sizeof(m_mt));
    }

protected:
    void _generate();

    uint32_t m_mt[kNumEntries]; ///< The random state vector
    int m_index;                ///< If set to >= kMaxEntries it means reset
};

typedef Mt19937RandomGenerator DefaultRandomGenerator;

} // namespace Slang

#endif // SLANG_RANDOM_GENERATOR_H
