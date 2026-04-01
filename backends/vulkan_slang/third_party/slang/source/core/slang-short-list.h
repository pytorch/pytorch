#ifndef SLANG_CORE_SHORT_LIST_H
#define SLANG_CORE_SHORT_LIST_H

#include "slang-allocator.h"
#include "slang-array-view.h"
#include "slang-math.h"
#include "slang.h"

namespace Slang
{
template<typename T, int shortListSize = 16, typename TAllocator = StandardAllocator>
class ShortList
{
private:
    static const Index kInitialCount = 16;
    typedef ShortList<T, shortListSize, TAllocator> ThisType;

public:
    ShortList()
        : m_buffer(nullptr), m_count(0), m_capacity(0)
    {
    }
    template<typename... Args>
    ShortList(const T& val, Args... args)
    {
        _init(val, args...);
    }
    ShortList(const ThisType& list)
        : m_buffer(nullptr), m_count(0), m_capacity(0)
    {
        this->operator=(list);
    }
    ShortList(ThisType&& list)
        : m_buffer(nullptr), m_count(0), m_capacity(0)
    {
        this->operator=(static_cast<ThisType&&>(list));
    }
    ~ShortList() { _deallocateBuffer(); }
    template<int _otherShortListSize, typename TOtherAllocator>
    ThisType& operator=(const ShortList<T, _otherShortListSize, TOtherAllocator>& list)
    {
        clearAndDeallocate();
        addRange(list);
        return *this;
    }

    ThisType& operator=(const ThisType& other)
    {
        clearAndDeallocate();
        addRange(other);
        return *this;
    }

    ThisType& operator=(ThisType&& list)
    {
        // Could just do a swap here, and memory would be freed on rhs dtor
        _deallocateBuffer();
        m_count = list.m_count;
        m_capacity = list.m_capacity;
        m_buffer = list.m_buffer;

        list.m_buffer = nullptr;
        list.m_count = 0;
        list.m_capacity = 0;

        for (Index i = 0; i < Math::Min((Index)shortListSize, m_count); i++)
            m_shortBuffer[i] = _Move(list.m_shortBuffer[i]);
        return *this;
    }

    struct Iterator
    {
        ThisType* container = nullptr;
        Index index = -1;
        Iterator& operator++()
        {
            ++index;
            return *this;
        }
        Iterator operator++(int)
        {
            Iterator rs = *this;
            ++index;
            return rs;
        }
        Iterator& operator--()
        {
            --index;
            return *this;
        }
        Iterator operator--(int)
        {
            Iterator rs = *this;
            --index;
            return rs;
        }
        T* operator->()
        {
            SLANG_ASSERT(container);
            return &(*container)[index];
        }
        T& operator*()
        {
            SLANG_ASSERT(container);
            return (*container)[index];
        }
        bool operator==(Iterator other)
        {
            return container == other.container && index == other.index;
        }
        bool operator!=(Iterator other)
        {
            return index != other.index || container != other.container;
        }
    };

    Iterator begin() const
    {
        Iterator rs;
        rs.container = const_cast<ThisType*>(this);
        rs.index = 0;
        return rs;
    }
    Iterator end() const
    {
        Iterator rs;
        rs.container = const_cast<ThisType*>(this);
        rs.index = m_count;
        return rs;
    }

    const T& getFirst() const
    {
        SLANG_ASSERT(m_count > 0);
        return m_shortBuffer[0];
    }

    T& getFirst()
    {
        SLANG_ASSERT(m_count > 0);
        return m_shortBuffer[0];
    }

    const T& getLast() const
    {
        SLANG_ASSERT(m_count > 0);
        if (m_count <= shortListSize)
            return m_shortBuffer[m_count - 1];
        return m_buffer[m_count - shortListSize - 1];
    }

    T& getLast()
    {
        SLANG_ASSERT(m_count > 0);
        if (m_count <= shortListSize)
            return m_shortBuffer[m_count - 1];
        return m_buffer[m_count - shortListSize - 1];
    }

    void removeLast()
    {
        SLANG_ASSERT(m_count > 0);
        m_count--;
    }

    struct GetArrayViewResult
    {
        ArrayView<T> arrayView;
        bool ownsStorage = false;

        GetArrayViewResult() = default;
        GetArrayViewResult(const GetArrayViewResult&) = delete;
        GetArrayViewResult(GetArrayViewResult&& other)
        {
            ownsStorage = other.ownsStorage;
            arrayView = other.arrayView;
            other.ownsStorage = false;
        }
        ~GetArrayViewResult()
        {
            if (ownsStorage)
            {
                ThisType::_free(arrayView.m_buffer, arrayView.m_count);
            }
        }
        T* getBuffer() { return arrayView.getBuffer(); }
    };
    inline GetArrayViewResult getArrayView() const
    {
        GetArrayViewResult result;
        if (m_count > shortListSize)
        {
            result.arrayView.m_buffer = ThisType::_allocate(m_count);
            result.arrayView.m_count = m_count;
            for (Index i = 0; i < shortListSize; i++)
                result.arrayView.m_buffer[i] = m_shortBuffer[i];
            for (Index i = shortListSize; i < m_count; i++)
                result.arrayView.m_buffer[i] = m_buffer[i - shortListSize];
            result.ownsStorage = true;
        }
        else
        {
            result.arrayView.m_buffer = const_cast<T*>(&m_shortBuffer[0]);
            result.arrayView.m_count = m_count;
        }
        return result;
    }

    inline GetArrayViewResult getArrayView(Index start, Index count) const
    {
        SLANG_ASSERT(start >= 0 && count >= 0 && start + count <= m_count);
        GetArrayViewResult result;
        if (start < shortListSize && start + count > shortListSize)
        {
            result.ownsStorage = true;
            result.arrayView.m_count = count;
            result.arrayView.m_buffer = ThisType::_allocate(count);
            for (Index i = start; i < shortListSize; i++)
                result.arrayView.m_buffer[i - start] = m_shortBuffer[i];
            for (Index i = shortListSize; i < start + count; i++)
                result.arrayView.m_buffer[i - start] = m_buffer[i - shortListSize];
            return result;
        }
        else if (start + count <= shortListSize)
        {
            result.ownsStorage = false;
            result.arrayView.m_count = count;
            result.arrayView.m_buffer = const_cast<T*>(m_shortBuffer) + start;
            return result;
        }
        else
        {
            result.ownsStorage = false;
            result.arrayView.m_count = count;
            result.arrayView.m_buffer = m_buffer + start - shortListSize;
            return result;
        }
    }

    void _maybeReserveForAdd()
    {
        if (m_capacity <= m_count - shortListSize)
        {
            Index newBufferSize = kInitialCount;
            if (m_capacity)
                newBufferSize = (m_capacity << 1);

            reserveOverflowBuffer(newBufferSize);
        }
    }

    void add(T&& obj)
    {
        if (m_count < shortListSize)
        {
            m_shortBuffer[m_count] = static_cast<T&&>(obj);
            m_count++;
            return;
        }
        _maybeReserveForAdd();
        m_buffer[m_count - shortListSize] = static_cast<T&&>(obj);
        m_count++;
    }

    void add(const T& obj)
    {
        if (m_count < shortListSize)
        {
            m_shortBuffer[m_count] = obj;
            m_count++;
            return;
        }
        _maybeReserveForAdd();
        m_buffer[m_count - shortListSize] = obj;
        m_count++;
    }

    Index getCount() const { return m_count; }

    void addRange(const T* vals, Index n)
    {
        for (Index i = 0; i < n; i++)
            add(vals[i]);
    }

    void addRange(ArrayView<T> list) { addRange(list.m_buffer, list.m_count); }

    void addRange(ConstArrayView<T> list) { addRange(list.m_buffer, list.m_count); }

    template<int _otherShortListSize, typename TOtherAllocator>
    void addRange(const ShortList<T, _otherShortListSize, TOtherAllocator>& list)
    {
        for (Index i = 0; i < list.getCount(); i++)
            add(list[i]);
    }

    void fastRemove(const T& val)
    {
        Index idx = indexOf(val);
        if (idx >= 0)
        {
            fastRemoveAt(idx);
        }
    }

    void fastRemoveAt(Index idx)
    {
        SLANG_ASSERT(idx >= 0 && idx < m_count);

        if (idx != m_count - 1)
        {
            (*this)[idx] = _Move(getLast());
        }
        m_count--;
    }

    void clear() { m_count = 0; }

    void clearAndDeallocate()
    {
        _deallocateBuffer();
        m_count = m_capacity = 0;
    }

    void reserveOverflowBuffer(Index size)
    {
        if (size > m_capacity)
        {
            T* newBuffer = _allocate(size);
            if (m_capacity)
            {
                for (Index i = 0; i < m_count - shortListSize; i++)
                    newBuffer[i] = static_cast<T&&>(m_buffer[i]);

                // Default-initialize the remaining elements
                for (Index i = m_count - shortListSize; i < size; i++)
                {
                    new (newBuffer + i) T();
                }
                _deallocateBuffer();
            }
            m_buffer = newBuffer;
            m_capacity = size;
        }
    }

    void setCount(Index count)
    {
        if (count > shortListSize)
            reserveOverflowBuffer(count - shortListSize);
        m_count = count;
    }

    void unsafeShrinkToCount(Index count) { m_count = count; }

    void compress()
    {
        if (m_capacity > m_count - shortListSize && m_count > shortListSize)
        {
            T* newBuffer = nullptr;
            if (m_count > shortListSize)
            {
                newBuffer = _allocate(m_count - shortListSize);
                for (Index i = shortListSize; i < m_count; i++)
                    newBuffer[i - shortListSize] = static_cast<T&&>(m_buffer[i - shortListSize]);
            }
            _deallocateBuffer();
            m_buffer = newBuffer;
            m_capacity = m_count - shortListSize;
        }
    }

    SLANG_FORCE_INLINE const T& operator[](Index idx) const
    {
        SLANG_ASSERT(idx >= 0 && idx < m_count);
        return (idx < shortListSize) ? m_shortBuffer[idx] : m_buffer[idx - shortListSize];
    }

    SLANG_FORCE_INLINE T& operator[](Index idx)
    {
        SLANG_ASSERT(idx >= 0 && idx < m_count);
        return (idx < shortListSize) ? m_shortBuffer[idx] : m_buffer[idx - shortListSize];
    }

    template<typename Func>
    Index findFirstIndex(const Func& predicate) const
    {
        for (Index i = 0; i < Math::Min(m_count, (Index)shortListSize); i++)
        {
            if (predicate(m_shortBuffer[i]))
                return i;
        }
        for (Index i = shortListSize; i < m_count; i++)
        {
            if (predicate(m_buffer[i - shortListSize]))
                return i;
        }
        return -1;
    }

    template<typename T2>
    Index indexOf(const T2& val) const
    {
        for (Index i = 0; i < Math::Min(m_count, (Index)shortListSize); i++)
        {
            if (m_shortBuffer[i] == val)
                return i;
        }
        for (Index i = shortListSize; i < m_count; i++)
        {
            if (m_buffer[i - shortListSize] == val)
                return i;
        }
        return -1;
    }

    template<typename Func>
    Index findLastIndex(const Func& predicate) const
    {
        for (Index i = m_count - 1; i >= shortListSize; i--)
        {
            if (predicate(m_buffer[i - shortListSize]))
                return i;
        }
        for (Index i = Math::Min((Index)shortListSize, m_count) - 1; i >= 0; i--)
        {
            if (predicate(m_shortBuffer[i]))
                return i;
        }
        return -1;
    }

    template<typename T2>
    Index lastIndexOf(const T2& val) const
    {
        for (Index i = m_count - 1; i >= shortListSize; i--)
        {
            if (m_buffer[i - shortListSize] == val)
                return i;
        }
        for (Index i = Math::Min((Index)shortListSize, m_count) - 1; i >= 0; i--)
        {
            if (m_shortBuffer[i] == val)
                return i;
        }
        return -1;
    }

    bool contains(const T& val) const { return indexOf(val) != Index(-1); }

    template<typename IterateFunc>
    void forEach(IterateFunc f) const
    {
        for (Index i = 0; i < m_count; i++)
            f(m_buffer[i]);
    }

private:
    T* m_buffer = nullptr; ///< A new T[N] allocated buffer. NOTE! All elements up to capacity are
                           ///< in some valid form for T.
    Index m_capacity = 0;  ///< The total capacity of elements in m_buffer
    Index m_count = 0;     ///< The amount of elements
    T m_shortBuffer[shortListSize];

    void _deallocateBuffer()
    {
        if (m_buffer)
        {
            AllocateMethod<T, TAllocator>::deallocateArray(m_buffer, m_capacity);
            m_buffer = nullptr;
        }
    }
    static inline T* _allocate(Index count)
    {
        return AllocateMethod<T, TAllocator>::allocateArray(count);
    }
    static inline void _free(T* ptr, Index count)
    {
        return AllocateMethod<T, TAllocator>::deallocateArray(ptr, count);
    }

    template<typename... Args>
    void _init(const T& val, Args... args)
    {
        add(val);
        _init(args...);
    }

    void _init() {}
};
} // namespace Slang

#endif
