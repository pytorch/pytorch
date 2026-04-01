#ifndef SLANG_CORE_STRING_SLICE_INDEX_MAP_H
#define SLANG_CORE_STRING_SLICE_INDEX_MAP_H

#include "slang-basic.h"
#include "slang-string-slice-pool.h"

namespace Slang
{

/* Maps an UnownedStringSlice to an index. All substrings are held internally in a StringSlicePool,
and so owned by the type.
*/
class StringSliceIndexMap
{
public:
    /// An index that identifies a key value pair.
    typedef Index CountIndex;

    /// Adds a key, value pair. Returns the CountIndex of the pair.
    /// If there is already a value stored for the key it is replaced.
    CountIndex add(const UnownedStringSlice& key, Index valueIndex);

    /// Finds or adds the slice. If the slice is added the defaultValueIndex is set.
    /// If not the index associated with the slice remains the same.
    /// Returns the CountIndex where the key,value pair are stored
    CountIndex findOrAdd(const UnownedStringSlice& key, Index defaultValueIndex);

    /// Gets the index associated with the key. Returns -1 if there is no associated index.
    SLANG_FORCE_INLINE Index getValue(const UnownedStringSlice& key);

    /// Get the amount of pairs in the map
    Index getCount() const { return m_indexMap.getCount(); }

    /// Get the slice and the index at the specified index
    SLANG_INLINE KeyValuePair<UnownedStringSlice, Index> getAt(CountIndex countIndex) const;

    /// Clear the contents of the map
    void clear();

    /// Get the key at the specified index
    UnownedStringSlice getKeyAt(CountIndex index) const
    {
        return m_pool.getSlice(StringSlicePool::Handle(index));
    }
    /// Get the value at the specified index
    Index& getValueAt(CountIndex index) { return m_indexMap[index]; }

    /// Get the amount of key,value pairs
    Index getCount() { return m_indexMap.getCount(); }

    /// Ctor
    StringSliceIndexMap()
        : m_pool(StringSlicePool::Style::Empty)
    {
    }

protected:
    StringSlicePool m_pool; ///< Pool holds the substrings
    List<Index> m_indexMap; ///< Maps a pool index to the output index
};

// ---------------------------------------------------------------------------
Index StringSliceIndexMap::getValue(const UnownedStringSlice& key)
{
    const Index poolIndex = m_pool.findIndex(key);
    return (poolIndex >= 0) ? m_indexMap[poolIndex] : -1;
}

// ---------------------------------------------------------------------------
KeyValuePair<UnownedStringSlice, Index> StringSliceIndexMap::getAt(CountIndex countIndex) const
{
    KeyValuePair<UnownedStringSlice, Index> pair;
    pair.key = m_pool.getSlice(StringSlicePool::Handle(countIndex));
    pair.value = m_indexMap[countIndex];
    return pair;
}

} // namespace Slang

#endif
