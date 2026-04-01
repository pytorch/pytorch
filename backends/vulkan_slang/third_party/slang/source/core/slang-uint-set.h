#ifndef SLANG_CORE_UINT_SET_H
#define SLANG_CORE_UINT_SET_H

#include "slang-common.h"
#include "slang-hash.h"
#include "slang-list.h"
#include "slang-math.h"

#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <memory.h>

namespace Slang
{

constexpr Index intLog2(unsigned x)
{
    return x == 1 ? 0 : 1 + intLog2(x >> 1);
}

// if `in` is 0, result is undefined behavior
static inline Index bitscanForward(uint64_t in)
{
    SLANG_ASSERT(in != 0);
#if defined(_MSC_VER)

#ifdef _WIN64
    uint64_t out = 0;
    _BitScanForward64((unsigned long*)&out, in);
    return Index(out);
#else
    uint32_t out;
    // check for 0s in 0bit->31bit. If all 0's, check for 0s in 32bit->63bit
    if (_BitScanForward((unsigned long*)&out, *(((uint32_t*)&in))))
        return Index(out);
    _BitScanForward((unsigned long*)&out, *(((uint32_t*)&in) + 1));
    return Index(out) + 32;
#endif // #ifdef _WIN64

#else
    return Index(__builtin_ctzll(in));
#endif // #if defined(_MSC_VER)
}

/* Hold a set of UInt values. Implementation works by storing as a bit per value */
/// UIntSet is essentially a Element[], where each Element is `b` bits big.
/// Each index has `b` number of integers. If the bit is 1, we have an element there.
/// Value of each element is equal to the binary offset from Element[0], bit 0.
class UIntSet
{
public:
    typedef UIntSet ThisType;
    typedef uint64_t Element; ///< Type that holds the bits to say if value is present

    constexpr static Index kElementSize =
        sizeof(Element) * 8; ///< The number of bits in an element. This also determines how many
                             ///< values a element can hold.
    constexpr static Index kElementMask = kElementSize - 1; ///< Mask to get shift from an index
    constexpr static Index kElementShift = intLog2(
        sizeof(Element) * 8); ///< How many bits to shift to get Element index from an index. 5 for
                              ///< 2^5=32 elements in a uint32_t. 6 for 2^6=64 in a uint64_t.

    UIntSet() {}
    UIntSet(const UIntSet& other) { m_buffer = other.m_buffer; }
    UIntSet(UIntSet&& other) { *this = (_Move(other)); }
    UIntSet(UInt maxVal) { resizeAndClear(maxVal); }

    UIntSet& operator=(UIntSet&& other);
    UIntSet& operator=(const UIntSet& other);

    HashCode getHashCode() const;

    /// Return the count of all bits directly represented
    Int getCount() const { return Int(m_buffer.getCount()) * kElementSize; }

    const List<Element>& getBuffer() const { return m_buffer; }

    /// Resize such that val can be stored and clear contents
    void resizeAndClear(UInt val);
    /// Set all of the values up to count, as set
    void setAll();
    /// Resize (but maintain contents) up to bit size.
    /// NOTE! That since storage is in Element blocks, it may mean some values after size are set
    /// (up to the Element boundary)
    void resize(UInt size);
    void resizeBackingBufferDirectly(Index size);

    /// Clear all of the contents (by clearing the bits)
    void clear();

    /// Clear all the contents and free memory
    void clearAndDeallocate();

    /// Add a value
    inline void add(UInt val);
    inline void add(const UIntSet& val);
    inline void addRange(const List<UInt>& other);

    inline void addRawElement(Element val, Index bitOffset);

    /// Remove a value
    inline void remove(UInt val);
    /// Returns true if the value is present
    inline bool contains(UInt val) const;

    inline bool contains(const UIntSet& set) const;

    /// ==
    bool operator==(const UIntSet& set) const;
    /// !=
    bool operator!=(const UIntSet& set) const { return !(*this == set); }

    /// Store the union between this and set
    void unionWith(const UIntSet& set);
    /// Store the intersection between this and set
    void intersectWith(const UIntSet& set);
    /// Store the subtraction between this and set
    void subtractWith(const UIntSet& set);

    ///
    bool isEmpty() const;

    /// Swap this with rhs
    void swapWith(ThisType& rhs) { m_buffer.swapWith(rhs.m_buffer); }

    template<typename T>
    List<T> getElements() const;
    Index countElements() const;

    /// Store the union of set1 and set2 in outRs
    static void calcUnion(UIntSet& outRs, const UIntSet& set1, const UIntSet& set2);
    /// Store the intersection of set1 and set2 in outRs
    static void calcIntersection(UIntSet& outRs, const UIntSet& set1, const UIntSet& set2);
    /// Store the subtraction of set2 from set1 in outRs
    static void calcSubtract(UIntSet& outRs, const UIntSet& set1, const UIntSet& set2);

    /// Returns true if set1 and set2 have a same value set (ie there is an intersection)
    static bool hasIntersection(const UIntSet& set1, const UIntSet& set2);

    /// Get LSB Zero of UIntSet. LSB Zero is the smallest value missing from this UIntSet.
    Index getLSBZero();

    struct Iterator
    {
        friend class UIntSet;

    private:
        const List<Element>* m_context;
        Index m_block = 0;
        Element m_processedElement = 0;
        uint64_t m_LSB = 0;

        void clearLSB()
        {
            m_LSB = bitscanForward(m_processedElement);
            m_processedElement &= m_processedElement - 1;
        }

        Iterator(const List<Element>* context) { m_context = context; }

    public:
        Element operator*() { return Element(m_LSB + (kElementSize * m_block)); }

        Iterator& operator++()
        {
            while (m_processedElement == 0)
            {
                m_block++;
                if (m_block >= m_context->getCount())
                {
                    return *this;
                }
                m_processedElement = (*m_context)[m_block];
            }
            clearLSB();
            return *this;
        }
        Iterator& operator++(int) { return ++(*this); }
        bool operator==(const Iterator& other) const
        {
            return other.m_block == this->m_block &&
                   other.m_processedElement == this->m_processedElement;
        }
        bool operator!=(const Iterator& other) const { return !(other == *this); }
    };
    Iterator begin() const
    {
        Iterator tmp(&m_buffer);
        if (m_buffer.getCount() == 0)
            return tmp;

        tmp.m_processedElement = m_buffer[0];
        if (tmp.m_processedElement == 0)
        {
            tmp++;
            return tmp;
        }

        tmp.clearLSB();
        return tmp;
    }
    Iterator end() const
    {
        Iterator tmp(&m_buffer);
        tmp.m_block = m_buffer.getCount();
        tmp.m_processedElement = 0;
        return tmp;
    }

    bool areAllZero() { return _areAllZero(m_buffer.getBuffer(), m_buffer.getCount()); }

protected:
    static bool _areAllZero(const UIntSet::Element* elems, Index count)
    {
        for (Index i = 0; i < count; ++i)
        {
            if (elems[i])
            {
                return false;
            }
        }
        return true;
    }

    List<Element> m_buffer;
};

// --------------------------------------------------------------------------
inline void UIntSet::remove(UInt val)
{
    const Index idx = Index(val >> kElementShift);
    if (idx < m_buffer.getCount())
    {
        m_buffer[idx] &= ~(Element(1) << (val & kElementMask));
    }
}

// --------------------------------------------------------------------------
inline bool UIntSet::contains(UInt val) const
{
    const Index idx = Index(val >> kElementShift);
    return idx < m_buffer.getCount() &&
           ((m_buffer[idx] & (Element(1) << (val & kElementMask))) != 0);
}

// --------------------------------------------------------------------------
inline bool UIntSet::contains(const UIntSet& set) const
{
    for (Index i = 0; i < set.m_buffer.getCount(); i++)
    {
        if (i >= m_buffer.getCount())
        {
            if (set.m_buffer[i])
                return false;
        }
        else
        {
            if ((m_buffer[i] & set.m_buffer[i]) != set.m_buffer[i])
                return false;
        }
    }
    return true;
}

// --------------------------------------------------------------------------

inline void UIntSet::resizeBackingBufferDirectly(Index newCount)
{
    const Index oldCount = m_buffer.getCount();
    m_buffer.setCount(newCount);

    if (newCount > oldCount)
    {
        ::memset(m_buffer.getBuffer() + oldCount, 0, (newCount - oldCount) * sizeof(Element));
    }
}

inline void UIntSet::add(UInt val)
{
    const Index idx = Index(val >> kElementShift);
    if (idx >= m_buffer.getCount())
    {
        resize(val + 1);
    }
    m_buffer[idx] |= Element(1) << (val & kElementMask);
}

inline void UIntSet::add(const UIntSet& other)
{
    auto otherCount = other.m_buffer.getCount();
    if (this->m_buffer.getCount() < otherCount)
        resizeBackingBufferDirectly(otherCount);

    for (auto i = 0; i < otherCount; i++)
        m_buffer[i] |= other.m_buffer[i];
}

inline void UIntSet::addRange(const List<UInt>& other)
{
    for (auto i : other)
        add(i);
}

inline void UIntSet::addRawElement(Element other, Index elementIndex)
{
    if (this->m_buffer.getCount() <= elementIndex)
        resizeBackingBufferDirectly(elementIndex + 1);
    m_buffer[elementIndex] |= other;
}

template<typename T>
List<T> UIntSet::getElements() const
{
    auto count = m_buffer.getCount();
    if (count == 0)
        return {};

    // Specific path for uint64_t. If using SIMD we should not use this path due to larger data
    // types.

    List<T> elements;
    elements.reserve(count);
    for (Index block = 0; block < count; block++)
    {
        Element n = m_buffer[block];
        while (n != 0)
        {
            elements.add(T(bitscanForward((uint64_t)n) + (kElementSize * block)));
            n &= n - 1;
        }
    }
    return elements;
}

} // namespace Slang
#endif
