#include "slang-string-slice-index-map.h"

namespace Slang
{

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! StringSliceIndexMap !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

StringSliceIndexMap::CountIndex StringSliceIndexMap::add(
    const UnownedStringSlice& key,
    Index valueIndex)
{
    StringSlicePool::Handle handle;
    m_pool.findOrAdd(key, handle);
    const CountIndex countIndex = StringSlicePool::asIndex(handle);
    if (countIndex >= m_indexMap.getCount())
    {
        SLANG_ASSERT(countIndex == m_indexMap.getCount());
        m_indexMap.add(valueIndex);
    }
    else
    {
        m_indexMap[countIndex] = valueIndex;
    }
    return countIndex;
}

StringSliceIndexMap::CountIndex StringSliceIndexMap::findOrAdd(
    const UnownedStringSlice& key,
    Index defaultValueIndex)
{
    StringSlicePool::Handle handle;
    m_pool.findOrAdd(key, handle);
    const CountIndex countIndex = StringSlicePool::asIndex(handle);
    if (countIndex >= m_indexMap.getCount())
    {
        SLANG_ASSERT(countIndex == m_indexMap.getCount());
        m_indexMap.add(defaultValueIndex);
    }
    return countIndex;
}

void StringSliceIndexMap::clear()
{
    m_pool.clear();
    m_indexMap.clear();
}

} // namespace Slang
