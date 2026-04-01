#ifndef SLANG_CORE_ARRAY_H
#define SLANG_CORE_ARRAY_H

#include "slang-array-view.h"
#include "slang-exception.h"

namespace Slang
{
/* An array container with fixed maximum size defined by COUNT. */
template<typename T, Index COUNT>
class Array
{
public:
    T* begin() { return m_buffer; }
    const T* begin() const { return m_buffer; }

    const T* end() const { return m_buffer + m_count; }
    T* end() { return m_buffer + m_count; }

    inline Index getCapacity() const { return COUNT; }
    inline Index getCount() const { return m_count; }
    inline const T& getFirst() const
    {
        SLANG_ASSERT(m_count > 0);
        return m_buffer[0];
    }
    inline T& getFirst()
    {
        SLANG_ASSERT(m_count > 0);
        return m_buffer[0];
    }
    inline const T& getLast() const
    {
        SLANG_ASSERT(m_count > 0);
        return m_buffer[m_count - 1];
    }
    inline T& getLast()
    {
        SLANG_ASSERT(m_count > 0);
        return m_buffer[m_count - 1];
    }
    inline void setCount(Index newCount)
    {
        SLANG_ASSERT(newCount >= 0 && newCount <= COUNT);
        m_count = newCount;
    }
    inline void add(const T& item)
    {
        SLANG_ASSERT(m_count < COUNT);
        m_buffer[m_count++] = item;
    }
    inline void add(T&& item)
    {
        SLANG_ASSERT(m_count < COUNT);
        m_buffer[m_count++] = _Move(item);
    }

    inline const T& operator[](Index idx) const
    {
        SLANG_ASSERT(idx >= 0 && idx < m_count);
        return m_buffer[idx];
    }
    inline T& operator[](Index idx)
    {
        SLANG_ASSERT(idx >= 0 && idx < m_count);
        return m_buffer[idx];
    }

    inline const T* getBuffer() const { return m_buffer; }
    inline T* getBuffer() { return m_buffer; }

    inline void clear() { m_count = 0; }

    template<typename T2>
    Index indexOf(const T2& val) const
    {
        return getView().indexOf(val);
    }
    template<typename T2>
    Index lastIndexOf(const T2& val) const
    {
        return getView().lastIndexOf(val);
    }
    template<typename Func>
    Index findFirstIndex(const Func& predicate) const
    {
        return getView().findFirstIndex(predicate);
    }
    template<typename Func>
    Index findLastIndex(const Func& predicate) const
    {
        return getView().findLastIndex(predicate);
    }

    inline ConstArrayView<T> getView() const { return ConstArrayView<T>(m_buffer, m_count); }
    inline ConstArrayView<T> getView(Index start, Index count) const
    {
        SLANG_ASSERT(start >= 0 && count >= 0);
        SLANG_ASSERT(start <= m_count && start + count < m_count);
        return ConstArrayView<T>(m_buffer + start, count);
    }

    inline ArrayView<T> getView() { return ArrayView<T>(m_buffer, m_count); }
    inline ArrayView<T> getView(Index start, Index count)
    {
        SLANG_ASSERT(start >= 0 && count >= 0);
        SLANG_ASSERT(start <= m_count && start + count < m_count);
        return ArrayView<T>(m_buffer + start, count);
    }

private:
    T m_buffer[COUNT];
    Index m_count = 0;
};

template<typename T>
class Array<T, 0>
{
public:
    T* begin() { return nullptr; }
    const T* begin() const { return nullptr; }

    const T* end() const { return nullptr; }
    T* end() { return nullptr; }

    inline Index getCapacity() const { return 0; }
    inline Index getCount() const { return 0; }
    inline void setCount(Index newCount) { SLANG_ASSERT(newCount == 0); }
    inline const T* getBuffer() const { return nullptr; }
    inline T* getBuffer() { return nullptr; }
    inline void clear() {}

    template<typename T2>
    Index indexOf(const T2& val) const
    {
        return getView().indexOf(val);
    }
    template<typename T2>
    Index lastIndexOf(const T2& val) const
    {
        return getView().lastIndexOf(val);
    }
    template<typename Func>
    Index findFirstIndex(const Func& predicate) const
    {
        return getView().findFirstIndex(predicate);
    }
    template<typename Func>
    Index findLastIndex(const Func& predicate) const
    {
        return getView().findLastIndex(predicate);
    }

    inline ConstArrayView<T> getView() const { return ConstArrayView<T>(nullptr, 0); }
    inline ConstArrayView<T> getView(Index start, Index count) const
    {
        SLANG_ASSERT(start == 0 && count == 0);
        return ConstArrayView<T>(nullptr, 0);
    }

    inline ArrayView<T> getView() { return ArrayView<T>(nullptr, 0); }
    inline ArrayView<T> getView(Index start, Index count)
    {
        SLANG_ASSERT(start == 0 && count == 0);
        return ArrayView<T>(nullptr, 0);
    }
};

template<typename T, typename... TArgs>
struct FirstType
{
    typedef T Type;
};


template<typename T, Index SIZE>
void insertArray(Array<T, SIZE>&)
{
}

template<typename T, typename... TArgs, Index SIZE>
void insertArray(Array<T, SIZE>& arr, const T& val, TArgs... args)
{
    arr.add(val);
    insertArray<T>(arr, args...);
}

template<typename... TArgs>
auto makeArray(TArgs... args) -> Array<typename FirstType<TArgs...>::Type, sizeof...(args)>
{
    Array<typename FirstType<TArgs...>::Type, Index(sizeof...(args))> rs;
    insertArray<typename FirstType<TArgs...>::Type>(rs, args...);
    return rs;
}

template<typename T>
auto makeArray() -> Array<T, 0>
{
    return Array<T, 0>();
}


template<typename TList>
void addToList(TList&)
{
}
template<typename TList, typename T>
void addToList(TList& list, T node)
{
    list.add(node);
}
template<typename TList, typename T, typename... TArgs>
void addToList(TList& list, T node, TArgs... args)
{
    list.add(node);
    addToList(list, args...);
}
} // namespace Slang

#endif
