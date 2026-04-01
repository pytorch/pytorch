#ifndef SLANG_CORE_LIST_H
#define SLANG_CORE_LIST_H

#include "slang-allocator.h"
#include "slang-array-view.h"
#include "slang-math.h"
#include "slang.h"

#include <algorithm>
#include <new>
#include <type_traits>


namespace Slang
{
// List is container of values of a type held consecutively in memory (much like std::vector)
//
// Note that in this implementation, the underlying memory is backed via an allocation of
// T[capacity] This means that all values have to be in a valid state *even if they are not used*
// (ie indices >= m_count must be valid)
//
// Also note this implementation does not necessarily 'initialize' an element which is no longer
// used, and this may lead to surprising behavior. Say the list contains a single smart pointer, and
// the last element is removed (say with removeLast). The smart pointer will *not* be released. The
// smart pointer will be released if the that index is used (via say an add) or the List goes out of
// scope.
template<typename T, typename TAllocator = StandardAllocator>
class List
{
private:
    static const Index kInitialCount = 16;

public:
    typedef List ThisType;

    List()
        : m_buffer(nullptr), m_count(0), m_capacity(0)
    {
    }
    template<typename... Args>
    List(const T& val, Args... args)
        : m_buffer(nullptr), m_count(0), m_capacity(0)
    {
        _init(val, args...);
    }
    List(const List<T>& list)
        : m_buffer(nullptr), m_count(0), m_capacity(0)
    {
        this->operator=(list);
    }
    List(List<T>&& list)
        : m_buffer(nullptr), m_count(0), m_capacity(0)
    {
        this->operator=(static_cast<List<T>&&>(list));
    }
    List(ArrayView<T> view)
        : List()
    {
        addRange(view);
    }
    static List<T> makeRepeated(const T& val, Index count)
    {
        List<T> rs;
        rs.setCount(count);
        for (Index i = 0; i < count; i++)
            rs[i] = val;
        return rs;
    }
    ~List() { _deallocateBuffer(); }
    List<T>& operator=(const List<T>& list)
    {
        clearAndDeallocate();
        addRange(list);
        return *this;
    }

    List<T>& operator=(List<T>&& list)
    {
        // Could just do a swap here, and memory would be freed on rhs dtor

        _deallocateBuffer();
        m_count = list.m_count;
        m_capacity = list.m_capacity;
        m_buffer = list.m_buffer;

        list.m_buffer = nullptr;
        list.m_count = 0;
        list.m_capacity = 0;
        return *this;
    }

    const T* begin() const { return m_buffer; }
    const T* end() const { return m_buffer + m_count; }

    T* begin() { return m_buffer; }
    T* end() { return m_buffer + m_count; }

    const T& getFirst() const
    {
        SLANG_ASSERT(m_count > 0);
        return m_buffer[0];
    }

    T& getFirst()
    {
        SLANG_ASSERT(m_count > 0);
        return m_buffer[0];
    }

    const T& getLast() const
    {
        SLANG_ASSERT(m_count > 0);
        return m_buffer[m_count - 1];
    }

    T& getLast()
    {
        SLANG_ASSERT(m_count > 0);
        return m_buffer[m_count - 1];
    }

    void removeLast()
    {
        SLANG_ASSERT(m_count > 0);
        m_count--;
    }

    inline void swapWith(List<T, TAllocator>& other)
    {
        T* buffer = m_buffer;
        m_buffer = other.m_buffer;
        other.m_buffer = buffer;

        auto bufferSize = m_capacity;
        m_capacity = other.m_capacity;
        other.m_capacity = bufferSize;

        auto count = m_count;
        m_count = other.m_count;
        other.m_count = count;
    }

    T* detachBuffer()
    {
        T* rs = m_buffer;
        m_buffer = nullptr;
        m_count = 0;
        m_capacity = 0;
        return rs;
    }
    void attachBuffer(T* buffer, Index count, Index capacity)
    {
        // Can only attach a buffer if there isn't a buffer already associated
        SLANG_ASSERT(m_buffer == nullptr);
        SLANG_ASSERT(count <= capacity);
        m_buffer = buffer;
        m_count = count;
        m_capacity = capacity;
    }

    inline ArrayView<T> getArrayView() const { return ArrayView<T>(m_buffer, m_count); }

    inline ArrayView<T> getArrayView(Index start, Index count) const
    {
        SLANG_ASSERT(start >= 0 && count >= 0 && start + count <= m_count);
        return ArrayView<T>(m_buffer + start, count);
    }

    void _maybeReserveForAdd()
    {
        if (m_capacity <= m_count)
        {
            Index newBufferSize = kInitialCount;
            if (m_capacity)
                newBufferSize = (m_capacity << 1);

            reserve(newBufferSize);
        }
    }

    void add(T&& obj)
    {
        _maybeReserveForAdd();
        m_buffer[m_count++] = static_cast<T&&>(obj);
    }

    void add(const T& obj)
    {
        _maybeReserveForAdd();
        m_buffer[m_count++] = obj;
    }

    Index getCount() const { return m_count; }
    Index getCapacity() const { return m_capacity; }

    const T* getBuffer() const { return m_buffer; }
    T* getBuffer() { return m_buffer; }

    bool operator==(const ThisType& rhs) const
    {
        if (&rhs == this)
        {
            return true;
        }
        const Index count = getCount();
        if (count != rhs.getCount())
        {
            return false;
        }
        for (Index i = 0; i < count; ++i)
        {
            if ((*this)[i] != rhs[i])
            {
                return false;
            }
        }
        return true;
    }
    SLANG_FORCE_INLINE bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }

    void insert(Index idx, const T& val) { insertRange(idx, &val, 1); }

    void insertRange(Index idx, const T* vals, Index n)
    {
        if (m_capacity < m_count + n)
        {
            Index newBufferCount = kInitialCount;
            while (newBufferCount < m_count + n)
                newBufferCount = newBufferCount << 1;

            T* newBuffer = _allocate(newBufferCount);
            if (m_capacity)
            {
                /*if (std::has_trivial_copy_assign<T>::value &&
                std::has_trivial_destructor<T>::value)
                {
                    memcpy(newBuffer, buffer, sizeof(T) * id);
                    memcpy(newBuffer + id + n, buffer + id, sizeof(T) * (_count - id));
                }
                else*/
                {
                    for (Index i = 0; i < idx; i++)
                        newBuffer[i] = m_buffer[i];
                    for (Index i = idx; i < m_count; i++)
                        newBuffer[i + n] = T(static_cast<T&&>(m_buffer[i]));
                }
                _deallocateBuffer();
            }
            m_buffer = newBuffer;
            m_capacity = newBufferCount;
        }
        else
        {
            /*if (std::has_trivial_copy_assign<T>::value && std::has_trivial_destructor<T>::value)
                memmove(buffer + id + n, buffer + id, sizeof(T) * (_count - id));
            else*/
            {
                for (Index i = m_count; i > idx; i--)
                    m_buffer[i + n - 1] = static_cast<T&&>(m_buffer[i - 1]);
            }
        }
        /*if (std::has_trivial_copy_assign<T>::value && std::has_trivial_destructor<T>::value)
            memcpy(buffer + id, vals, sizeof(T) * n);
        else*/
        for (Index i = 0; i < n; i++)
            m_buffer[idx + i] = vals[i];

        m_count += n;
    }

    void insertRange(Index id, const List<T>& list)
    {
        insertRange(id, list.m_buffer, list.m_count);
    }

    void addRange(ArrayView<T> list) { insertRange(m_count, list.getBuffer(), list.getCount()); }

    void addRange(const T* vals, Index n) { insertRange(m_count, vals, n); }

    void addRange(const List<T>& list) { insertRange(m_count, list.m_buffer, list.m_count); }

    void removeRange(Index idx, Index count)
    {
        SLANG_ASSERT(idx >= 0 && idx <= m_count);

        const Index actualDeleteCount = ((idx + count) >= m_count) ? (m_count - idx) : count;
        for (Index i = idx + actualDeleteCount; i < m_count; i++)
            m_buffer[i - actualDeleteCount] = static_cast<T&&>(m_buffer[i]);
        m_count -= actualDeleteCount;
    }

    void removeAt(Index id) { removeRange(id, 1); }

    void remove(const T& val)
    {
        Index idx = indexOf(val);
        if (idx != -1)
            removeAt(idx);
    }

    void reverse()
    {
        for (Index i = 0; i < (m_count >> 1); i++)
        {
            swapElements(m_buffer, i, m_count - i - 1);
        }
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
        // We do not test for idx == m_count - 1 (ie the move is to current index). With the
        // assumption that any reasonable move implementation tests and ignores this case
        if (idx != m_count - 1)
        {
            m_buffer[idx] = _Move(m_buffer[m_count - 1]);
        }
        m_count--;
    }

    void clear() { m_count = 0; }

    void clearAndDeallocate()
    {
        _deallocateBuffer();
        m_count = m_capacity = 0;
    }

    void reserve(Index size)
    {
        // The cast for this comparison is needed, otherwise some compilers erroneously detect
        // the possiblity of a zero sized allocation (possible if m_capacity is assumed to be
        // negative).
        if (UIndex(size) > UIndex(m_capacity))
        {
            T* newBuffer = _allocate(size);
            if (m_capacity)
            {
                /*if (std::has_trivial_copy_assign<T>::value &&
                std::has_trivial_destructor<T>::value) memcpy(newBuffer, buffer, _count *
                sizeof(T)); else*/
                {
                    for (Index i = 0; i < m_count; i++)
                        newBuffer[i] = static_cast<T&&>(m_buffer[i]);

                    // Default-initialize the remaining elements
                    for (Index i = m_count; i < size; i++)
                    {
                        new (newBuffer + i) T();
                    }
                }
                _deallocateBuffer();
            }
            m_buffer = newBuffer;
            m_capacity = size;
        }
    }

    void growToCount(Index count)
    {
        Index newBufferCount = Index(1) << Math::Log2Ceil((unsigned int)count);
        if (m_capacity < newBufferCount)
        {
            reserve(newBufferCount);
        }
        m_count = count;
    }

    void setCount(Index count)
    {
        reserve(count);
        m_count = count;
    }

    void unsafeShrinkToCount(Index count) { m_count = count; }

    void compress()
    {
        if (m_capacity > m_count && m_count > 0)
        {
            T* newBuffer = _allocate(m_count);
            for (Index i = 0; i < m_count; i++)
                newBuffer[i] = static_cast<T&&>(m_buffer[i]);

            _deallocateBuffer();
            m_buffer = newBuffer;
            m_capacity = m_count;
        }
    }

    SLANG_FORCE_INLINE const T& operator[](Index idx) const
    {
        SLANG_ASSERT(idx >= 0 && idx < m_count);
        return m_buffer[idx];
    }

    SLANG_FORCE_INLINE T& operator[](Index idx)
    {
        SLANG_ASSERT(idx >= 0 && idx < m_count);
        return m_buffer[idx];
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

    bool contains(const T& val) const { return indexOf(val) != Index(-1); }

    void sort()
    {
        sort([](const T& t1, const T& t2) { return t1 < t2; });
    }

    template<typename Comparer>
    void sort(Comparer compare)
    {
        // insertionSort(buffer, 0, _count - 1);
        // quickSort(buffer, 0, _count - 1, compare);
        std::sort(m_buffer, m_buffer + m_count, compare);
    }

    void stableSort()
    {
        stableSort([](const T& t1, const T& t2) { return t1 < t2; });
    }

    template<typename Comparer>
    void stableSort(Comparer compare)
    {
        std::stable_sort(m_buffer, m_buffer + m_count, compare);
    }

    template<typename IterateFunc>
    void forEach(IterateFunc f) const
    {
        for (Index i = 0; i < m_count; i++)
            f(m_buffer[i]);
    }

    template<typename Comparer>
    void quickSort(T* vals, Index startIndex, Index endIndex, Comparer comparer)
    {
        static const Index kMinQSortSize = 32;

        if (startIndex < endIndex)
        {
            if (endIndex - startIndex < kMinQSortSize)
                insertionSort(vals, startIndex, endIndex, comparer);
            else
            {
                Index pivotIndex = (startIndex + endIndex) >> 1;
                Index pivotNewIndex = partition(vals, startIndex, endIndex, pivotIndex, comparer);
                quickSort(vals, startIndex, pivotNewIndex - 1, comparer);
                quickSort(vals, pivotNewIndex + 1, endIndex, comparer);
            }
        }
    }
    template<typename Comparer>
    Index partition(T* vals, Index left, Index right, Index pivotIndex, Comparer comparer)
    {
        T pivotValue = vals[pivotIndex];
        swapElements(vals, right, pivotIndex);
        Index storeIndex = left;
        for (Index i = left; i < right; i++)
        {
            if (comparer(vals[i], pivotValue))
            {
                swapElements(vals, i, storeIndex);
                storeIndex++;
            }
        }
        swapElements(vals, storeIndex, right);
        return storeIndex;
    }
    template<typename Comparer>
    void insertionSort(T* vals, Index startIndex, Index endIndex, Comparer comparer)
    {
        for (Index i = startIndex + 1; i <= endIndex; i++)
        {
            T insertValue = static_cast<T&&>(vals[i]);
            Index insertIndex = i - 1;
            while (insertIndex >= startIndex && comparer(insertValue, vals[insertIndex]))
            {
                vals[insertIndex + 1] = static_cast<T&&>(vals[insertIndex]);
                insertIndex--;
            }
            vals[insertIndex + 1] = static_cast<T&&>(insertValue);
        }
    }

    inline static void swapElements(T* vals, Index index1, Index index2)
    {
        if (index1 != index2)
        {
            T tmp = static_cast<T&&>(vals[index1]);
            vals[index1] = static_cast<T&&>(vals[index2]);
            vals[index2] = static_cast<T&&>(tmp);
        }
    }

    inline void swapElements(Index index1, Index index2) { swapElements(m_buffer, index1, index2); }

    template<typename T2, typename Comparer>
    Index binarySearch(const T2& obj, Comparer comparer) const
    {
        Index imin = 0, imax = m_count - 1;
        while (imax >= imin)
        {
            Index imid = imin + ((imax - imin) >> 1);
            int compareResult = comparer(m_buffer[imid], obj);
            if (compareResult == 0)
                return imid;
            else if (compareResult < 0)
                imin = imid + 1;
            else
                imax = imid - 1;
        }
        // TODO: The return value on a failed search should be
        // the bitwise negation of the index where `obj` should
        // be inserted to be in the proper sorted location.
        return -1;
    }

    template<typename T2>
    Index binarySearch(const T2& obj)
    {
        return binarySearch(
            obj,
            [](T& curObj, const T2& thatObj) -> int
            {
                if (curObj < thatObj)
                    return -1;
                else if (curObj == thatObj)
                    return 0;
                else
                    return 1;
            });
    }

private:
    T* m_buffer; ///< A new T[N] allocated buffer. NOTE! All elements up to capacity are in some
                 ///< valid form for T.
    Index m_capacity; ///< The total capacity of elements
    Index m_count;    ///< The amount of elements

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
    static void _free(T* buffer, Index count)
    {
        return AllocateMethod<T, TAllocator>::deallocateArray(buffer, count);
    }

    template<typename... Args>
    void _init(const T& val, Args... args)
    {
        add(val);
        _init(args...);
    }

    void _init() {}
};

template<typename T>
T calcMin(const List<T>& list)
{
    T minVal = list.getFirst();
    for (Index i = 1; i < list.getCount(); i++)
        if (list[i] < minVal)
            minVal = list[i];
    return minVal;
}

template<typename T>
T calcMax(const List<T>& list)
{
    T maxVal = list.getFirst();
    for (Index i = 1; i < list.getCount(); i++)
        if (list[i] > maxVal)
            maxVal = list[i];
    return maxVal;
}
} // namespace Slang

#endif
