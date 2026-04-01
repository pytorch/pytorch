#ifndef SLANG_CORE_CHUNKED_LIST_H
#define SLANG_CORE_CHUNKED_LIST_H

#include "slang-allocator.h"
#include "slang-array-view.h"
#include "slang-math.h"
#include "slang.h"

namespace Slang
{
// Items stored in a ChunkedList are guaranteed to have fixed address.
template<typename T, uint32_t defaultChunkSize = 16, typename TAllocator = StandardAllocator>
class ChunkedList
{
private:
    TAllocator allocator;

    struct Chunk
    {
        uint32_t size = 0;
        uint32_t capacity = defaultChunkSize;
        Chunk* next = nullptr;
        T* begin() { return reinterpret_cast<T*>(this + 1); }
        T* end() { return begin() + size; }
    };

    struct FirstChunk : public Chunk
    {
        T elements[defaultChunkSize];
    };

    Chunk* allocateChunk(uint32_t size)
    {
        auto resultChunk = (Chunk*)allocator.allocate(sizeof(Chunk) + size * sizeof(T));
        resultChunk->capacity = size;
        resultChunk->size = 0;
        resultChunk->next = nullptr;
        auto firstItem = resultChunk->begin();
        if (!std::is_trivially_constructible_v<T>)
        {
            for (uint32_t i = 0; i < size; i++)
                new (firstItem + i) T();
        }
        return resultChunk;
    }
    void freeChunk(Chunk* chunk)
    {
        if (!std::is_trivially_destructible_v<T>)
        {
            for (uint32_t i = 0; i < chunk->capacity; i++)
                chunk->begin()[i].~T();
        }
        allocator.deallocate(chunk);
    }

public:
    typedef ChunkedList<T, defaultChunkSize, TAllocator> ThisType;
    ChunkedList()
        : m_lastChunk(&m_firstChunk), m_count(0)
    {
    }
    template<typename... Args>
    ChunkedList(const T& val, Args... args)
    {
        _init(val, args...);
    }
    ChunkedList(const ThisType& list)
        : m_lastChunk(&m_firstChunk), m_count(0)
    {
        this->operator=(list);
    }
    ChunkedList(ThisType&& list)
        : m_lastChunk(&m_firstChunk), m_count(0)
    {
        this->operator=(static_cast<ThisType&&>(list));
    }
    ~ChunkedList() { _deallocateBuffer(); }
    template<int _otherShortListSize, typename TOtherAllocator>
    ThisType& operator=(const ChunkedList<T, _otherShortListSize, TOtherAllocator>& list)
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
        m_firstChunk = _Move(list.m_firstChunk);
        m_lastChunk = list.m_lastChunk;
        list.m_count = 0;
        list.m_firstChunk.next = nullptr;
        list.m_lastChunk = &list.m_firstChunk;
        list.m_firstChunk.size = 0;
        return *this;
    }

    struct Iterator
    {
        Chunk* chunk = nullptr;
        Index index = -1;
        Iterator& operator++()
        {
            ++index;
            if (index == chunk->size)
            {
                index = 0;
                chunk = chunk->next;
            }
            return *this;
        }
        Iterator operator++(int)
        {
            Iterator rs = *this;
            operator++();
            return rs;
        }
        T* operator->()
        {
            SLANG_ASSERT(chunk);
            return chunk->begin() + index;
        }
        T& operator*()
        {
            SLANG_ASSERT(chunk);
            return chunk->begin()[index];
        }
        bool operator==(Iterator other) { return chunk == other.chunk && index == other.index; }
        bool operator!=(Iterator other) { return index != other.index || chunk != other.chunk; }
    };

    Iterator begin()
    {
        Iterator rs;
        rs.chunk = &m_firstChunk;
        rs.index = 0;
        return rs;
    }
    Iterator end()
    {
        Iterator rs;
        rs.chunk = nullptr;
        rs.index = 0;
        return rs;
    }

    Chunk* _maybeReserveForAdd(uint32_t chunkSize)
    {
        if (m_lastChunk->capacity - m_lastChunk->size < chunkSize)
        {
            auto chunk = allocateChunk(Math::Max(defaultChunkSize, chunkSize));
            m_lastChunk->next = chunk;
            m_lastChunk = chunk;
            return chunk;
        }
        return m_lastChunk;
    }

    T* add(T&& obj)
    {
        auto chunk = _maybeReserveForAdd(1);
        auto result = chunk->begin() + chunk->size;
        chunk->begin()[chunk->size] = static_cast<T&&>(obj);
        chunk->size++;
        m_count++;
        return result;
    }

    T* add(const T& obj)
    {
        auto chunk = _maybeReserveForAdd(1);
        auto result = chunk->begin() + chunk->size;
        chunk->begin()[chunk->size] = obj;
        chunk->size++;
        m_count++;
        return result;
    }

    Index getCount() const { return m_count; }

    T* addRange(const T* vals, Index n)
    {
        Chunk* chunk = _maybeReserveForAdd((uint32_t)n);
        auto result = chunk->begin() + chunk->size;
        for (Index i = 0; i < n; i++)
        {
            chunk->begin()[chunk->size + i] = vals[i];
        }
        chunk->size += (uint32_t)n;
        m_count += n;
        return result;
    }

    T* addRange(ArrayView<T> list) { return addRange(list.m_buffer, list.m_count); }

    T* reserveRange(uint32_t size)
    {
        Chunk* chunk = _maybeReserveForAdd((uint32_t)size);
        auto result = chunk->begin() + chunk->size;
        chunk->size += size;
        m_count += size;
        return result;
    }

    template<typename TContainer>
    T* addRange(const TContainer& list)
    {
        Chunk* chunk = _maybeReserveForAdd((uint32_t)list.getCount());
        auto result = chunk->begin() + chunk->size;
        for (auto& obj : list)
        {
            chunk->begin()[chunk->size] = obj;
            chunk->size++;
            m_count++;
        }
        return result;
    }

    void clearAndDeallocate()
    {
        _deallocateBuffer();
        m_count = 0;
        for (auto& item : m_firstChunk.elements)
            item = T();
    }

private:
    Index m_count = 0; ///< The amount of elements
    FirstChunk m_firstChunk;
    Chunk* m_lastChunk = &m_firstChunk;

    void _deallocateBuffer()
    {
        auto chunk = m_firstChunk.next;
        while (chunk)
        {
            auto nextChunk = chunk->next;
            freeChunk(chunk);
            chunk = nextChunk;
        }
        m_firstChunk.next = 0;
        m_firstChunk.size = 0;
        m_lastChunk = &m_firstChunk;
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
