#include "slang-uint-set.h"

namespace Slang
{

Index UIntSet::getLSBZero()
{
    uint64_t offset = 0;
    for (Element& element : this->m_buffer)
    {
        // Flip all bits so bitscanForward can find a 0 bit
        Element flippedElement = ~element;

        // continue if we don't have 0 bits
        if (flippedElement == 0)
        {
            offset += sizeof(Element) * 8;
            continue;
        }

        // Get LSBZero of current Block, add with offset
        return bitscanForward(flippedElement) + offset;
    }
    return offset;
}

UIntSet& UIntSet::operator=(UIntSet&& other)
{
    m_buffer = _Move(other.m_buffer);
    return *this;
}

UIntSet& UIntSet::operator=(const UIntSet& other)
{
    m_buffer = other.m_buffer;
    return *this;
}

HashCode UIntSet::getHashCode() const
{
    int rs = 0;
    for (auto val : m_buffer)
        rs ^= val;
    return rs;
}

void UIntSet::resizeAndClear(UInt val)
{
    // TODO(JS): This could be faster in that if the resize is larger the additional area is cleared
    // twice
    resize(val);
    clear();
}

void UIntSet::setAll()
{
    ::memset(m_buffer.getBuffer(), -1, m_buffer.getCount() * sizeof(Element));
}

void UIntSet::resize(UInt size)
{
    const Index newCount = Index((size + kElementMask) >> kElementShift);
    resizeBackingBufferDirectly(newCount);
}

void UIntSet::clear()
{
    ::memset(m_buffer.getBuffer(), 0, m_buffer.getCount() * sizeof(Element));
}

bool UIntSet::isEmpty() const
{
    return _areAllZero(m_buffer.getBuffer(), m_buffer.getCount());
}

void UIntSet::clearAndDeallocate()
{
    m_buffer.clearAndDeallocate();
}

void UIntSet::unionWith(const UIntSet& set)
{
    const Index minCount = Math::Min(set.m_buffer.getCount(), m_buffer.getCount());
    for (Index i = 0; i < minCount; i++)
    {
        m_buffer[i] |= set.m_buffer[i];
    }

    if (set.m_buffer.getCount() > m_buffer.getCount())
        m_buffer.addRange(
            set.m_buffer.getBuffer() + m_buffer.getCount(),
            set.m_buffer.getCount() - m_buffer.getCount());
}

bool UIntSet::operator==(const UIntSet& set) const
{
    const Index aCount = m_buffer.getCount();
    const auto aElems = m_buffer.getBuffer();

    const Index bCount = set.m_buffer.getCount();
    const auto bElems = set.m_buffer.getBuffer();

    const Index minCount = Math::Min(aCount, bCount);

    return ::memcmp(aElems, bElems, minCount * sizeof(Element)) == 0 &&
           _areAllZero(aElems + minCount, aCount - minCount) &&
           _areAllZero(bElems + minCount, bCount - minCount);
}

void UIntSet::intersectWith(const UIntSet& set)
{
    if (set.m_buffer.getCount() < m_buffer.getCount())
        ::memset(
            m_buffer.getBuffer() + set.m_buffer.getCount(),
            0,
            (m_buffer.getCount() - set.m_buffer.getCount()) * sizeof(Element));

    const Index minCount = Math::Min(set.m_buffer.getCount(), m_buffer.getCount());
    for (Index i = 0; i < minCount; i++)
    {
        m_buffer[i] &= set.m_buffer[i];
    }
}

void UIntSet::subtractWith(const UIntSet& set)
{
    const Index minCount = Math::Min(this->m_buffer.getCount(), set.m_buffer.getCount());
    for (Index i = 0; i < minCount; i++)
    {
        this->m_buffer[i] = this->m_buffer[i] & (~set.m_buffer[i]);
    }
}

/* static */ void UIntSet::calcUnion(UIntSet& outRs, const UIntSet& set1, const UIntSet& set2)
{
    outRs.resizeBackingBufferDirectly(
        Math::Max(set1.m_buffer.getCount(), set2.m_buffer.getCount()));
    outRs.clear();
    for (Index i = 0; i < set1.m_buffer.getCount(); i++)
        outRs.m_buffer[i] |= set1.m_buffer[i];
    for (Index i = 0; i < set2.m_buffer.getCount(); i++)
        outRs.m_buffer[i] |= set2.m_buffer[i];
}

/* static */ void UIntSet::calcIntersection(
    UIntSet& outRs,
    const UIntSet& set1,
    const UIntSet& set2)
{
    const Index minCount = Math::Min(set1.m_buffer.getCount(), set2.m_buffer.getCount());
    outRs.resizeBackingBufferDirectly(minCount);

    for (Index i = 0; i < minCount; i++)
        outRs.m_buffer[i] = set1.m_buffer[i] & set2.m_buffer[i];
}

/* static */ void UIntSet::calcSubtract(UIntSet& outRs, const UIntSet& set1, const UIntSet& set2)
{
    const auto set1Count = set1.m_buffer.getCount();
    const auto set2Count = set2.m_buffer.getCount();

    outRs.resizeBackingBufferDirectly(set1Count);

    for (Index i = 0; i < set1Count; i++)
    {
        if (i < set2Count)
        {
            outRs.m_buffer[i] = set1.m_buffer[i] & (~set2.m_buffer[i]);
        }
        else
        {
            // If `set2` is smaller, copy the remaining values from `set1`
            outRs.m_buffer[i] = set1.m_buffer[i];
        }
    }
}

/* static */ bool UIntSet::hasIntersection(const UIntSet& set1, const UIntSet& set2)
{
    const Index minCount = Math::Min(set1.m_buffer.getCount(), set2.m_buffer.getCount());
    for (Index i = 0; i < minCount; i++)
    {
        if (set1.m_buffer[i] & set2.m_buffer[i])
            return true;
    }
    return false;
}

Index UIntSet::countElements() const
{
    // TODO: This can be made faster using SIMD intrinsics to count set bits.
    uint64_t tmp;
    constexpr Index loopSize =
        ((sizeof(Element) / sizeof(tmp)) != 0) ? sizeof(Element) / sizeof(tmp) : 1;
    Index count = 0;
    for (auto index = 0; index < this->m_buffer.getCount(); index++)
    {
        for (auto i = 0; i < loopSize; i++)
        {
            tmp = m_buffer[index] >> (sizeof(tmp) * i);
            tmp = tmp - ((tmp >> 1) & 0x5555555555555555);
            tmp = (tmp & 0x3333333333333333) + ((tmp >> 2) & 0x3333333333333333);
            count += ((tmp + (tmp >> 4) & 0xF0F0F0F0F0F0F0F) * 0x101010101010101) >> 56;
        }
    }
    return count;
}

} // namespace Slang
