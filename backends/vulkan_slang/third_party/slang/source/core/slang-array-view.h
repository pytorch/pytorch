#ifndef SLANG_CORE_ARRAY_VIEW_H
#define SLANG_CORE_ARRAY_VIEW_H

#include "slang-common.h"

namespace Slang
{

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!! ConstArrayView !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

template<typename T>
class ConstArrayView
{
public:
    typedef ConstArrayView ThisType;

    SLANG_FORCE_INLINE const T* begin() const { return m_buffer; }

    SLANG_FORCE_INLINE const T* end() const { return m_buffer + m_count; }

    SLANG_FORCE_INLINE Count getCount() const { return m_count; }

    SLANG_FORCE_INLINE const T& operator[](Index idx) const
    {
        SLANG_ASSERT(idx >= 0 && idx < m_count);
        return m_buffer[idx];
    }

    SLANG_FORCE_INLINE const T* getBuffer() const { return m_buffer; }

    template<typename T2>
    Index indexOf(const T2& val) const
    {
        for (Index i = 0; i < m_count; i++)
        {
            if (m_buffer[i] == val)
                return i;
        }
        return -1;
    }

    template<typename T2>
    Index lastIndexOf(const T2& val) const
    {
        for (Index i = m_count - 1; i >= 0; i--)
        {
            if (m_buffer[i] == val)
                return i;
        }
        return -1;
    }

    template<typename Func>
    Index findFirstIndex(const Func& predicate) const
    {
        for (Index i = 0; i < m_count; i++)
        {
            if (predicate(m_buffer[i]))
                return i;
        }
        return -1;
    }

    template<typename Func>
    Index findLastIndex(const Func& predicate) const
    {
        for (Index i = m_count - 1; i >= 0; i--)
        {
            if (predicate(m_buffer[i]))
                return i;
        }
        return -1;
    }

    bool containsMemory(const ThisType& rhs) const
    {
        return rhs.getBuffer() >= getBuffer() && rhs.end() <= end();
    }

    bool operator==(const ThisType& rhs) const
    {
        if (&rhs == this)
        {
            return true;
        }
        const Count count = getCount();
        if (count != rhs.getCount())
        {
            return false;
        }
        const T* thisEle = getBuffer();
        const T* rhsEle = rhs.getBuffer();
        for (Index i = 0; i < count; ++i)
        {
            if (thisEle[i] != rhsEle[i])
            {
                return false;
            }
        }
        return true;
    }
    SLANG_FORCE_INLINE bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }

    ThisType head(Index index) const
    {
        SLANG_ASSERT(index >= 0 && index <= m_count);
        return ThisType(m_buffer, index);
    }
    ThisType tail(Index index) const
    {
        SLANG_ASSERT(index >= 0 && index <= m_count);
        return ThisType(m_buffer + index, m_count - index);
    }

    ConstArrayView()
        : m_buffer(nullptr), m_count(0)
    {
    }

    ConstArrayView(const T* buffer, Count count)
        : m_buffer(const_cast<T*>(buffer)), m_count(count)
    {
    }

protected:
    ConstArrayView(T* buffer, Count count)
        : m_buffer(buffer), m_count(count)
    {
    }

    T* m_buffer; ///< Note that this isn't const, as is used for derived class ArrayView also
    Count m_count;
};

template<typename T>
ConstArrayView<T> makeConstArrayViewSingle(const T& obj)
{
    return ConstArrayView<T>(&obj, 1);
}

template<typename T>
ConstArrayView<T> makeConstArrayView(const T* buffer, Count count)
{
    return ConstArrayView<T>(buffer, count);
}

template<typename T, size_t N>
ConstArrayView<T> makeConstArrayView(const T (&arr)[N])
{
    return ConstArrayView<T>(arr, Index(N));
}


// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ArrayView !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

template<typename T>
class ArrayView : public ConstArrayView<T>
{
public:
    typedef ArrayView ThisType;

    typedef ConstArrayView<T> Super;

    using Super::m_buffer;
    using Super::m_count;

    using Super::begin;
    T* begin() { return m_buffer; }

    using Super::end;
    T* end() { return m_buffer + m_count; }

    using Super::head;
    using Super::tail;

    using Super::operator[];
    inline T& operator[](Index idx)
    {
        SLANG_ASSERT(idx >= 0 && idx < m_count);
        return m_buffer[idx];
    }

    using Super::getBuffer;
    inline T* getBuffer() { return m_buffer; }

    ThisType head(Index index)
    {
        SLANG_ASSERT(index >= 0 && index <= m_count);
        return ThisType(m_buffer, index);
    }
    ThisType tail(Index index)
    {
        SLANG_ASSERT(index >= 0 && index <= m_count);
        return ThisType(m_buffer + index, m_count - index);
    }

    T& getLast() { return m_buffer[m_count - 1]; }

    ArrayView()
        : Super()
    {
    }
    ArrayView(T* buffer, Index size)
        : Super(buffer, size)
    {
    }
};

template<typename T>
ArrayView<T> makeArrayViewSingle(T& obj)
{
    return ArrayView<T>(&obj, 1);
}

template<typename T>
ArrayView<T> makeArrayView(T* buffer, Count count)
{
    return ArrayView<T>(buffer, count);
}

template<typename T, size_t N>
ArrayView<T> makeArrayView(T (&arr)[N])
{
    return ArrayView<T>(arr, Count(N));
}


} // namespace Slang

#endif
