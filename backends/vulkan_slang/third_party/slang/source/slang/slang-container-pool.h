#ifndef SLANG_CONTAINER_POOL_H
#define SLANG_CONTAINER_POOL_H

#include "../core/slang-dictionary.h"
#include "../core/slang-list.h"
#include "../core/slang-virtual-object-pool.h"

// A pool to allow reuse of common types of containers to avoid
// frequent resizing and rehashing.

namespace Slang
{
static const int kContainerPoolSize = 1024;

template<typename T>
struct ObjectPool
{
    ObjectPool(int maxElementCount)
    {
        m_pool.initPool(maxElementCount);
        m_objects.setCount(maxElementCount);
    }

    T* getObject()
    {
        auto id = m_pool.alloc(1);
        if (id == -1)
            SLANG_UNEXPECTED("container pool allocation failure.");
        return &m_objects[id];
    }

    void freeObject(T* object)
    {
        auto id = (int)(object - m_objects.getBuffer());
        m_pool.free(id, 1);
    }

    VirtualObjectPool m_pool;
    List<T> m_objects;
};

struct ContainerPool
{
    ObjectPool<List<void*>> m_listPool;
    ObjectPool<Dictionary<void*, void*>> m_dictionaryPool;
    ObjectPool<HashSet<void*>> m_hashSetPool;

    ContainerPool()
        : m_listPool(kContainerPoolSize)
        , m_dictionaryPool(kContainerPoolSize)
        , m_hashSetPool(kContainerPoolSize)
    {
    }

    template<typename T>
    List<T*>* getList()
    {
        return (List<T*>*)m_listPool.getObject();
    }

    template<typename T, typename U>
    Dictionary<T*, U*>* getDictionary()
    {
        return (Dictionary<T*, U*>*)m_dictionaryPool.getObject();
    }

    template<typename T>
    HashSet<T*>* getHashSet()
    {
        return (HashSet<T*>*)m_hashSetPool.getObject();
    }

    template<typename T>
    void free(List<T*>* list)
    {
        list->clear();
        m_listPool.freeObject((List<void*>*)list);
    }

    template<typename T, typename U>
    void free(Dictionary<T*, U*>* dict)
    {
        dict->clear();
        m_dictionaryPool.freeObject((Dictionary<void*, void*>*)dict);
    }

    template<typename T>
    void free(HashSet<T*>* set)
    {
        set->clear();
        m_hashSetPool.freeObject((HashSet<void*>*)set);
    }
};
} // namespace Slang

#endif
