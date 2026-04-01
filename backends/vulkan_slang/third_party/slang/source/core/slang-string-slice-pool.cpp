#include "slang-string-slice-pool.h"

namespace Slang
{

/* static */ const StringSlicePool::Handle StringSlicePool::kNullHandle;
/* static */ const StringSlicePool::Handle StringSlicePool::kEmptyHandle;

/* static */ const Index StringSlicePool::kDefaultHandlesCount;

StringSlicePool::StringSlicePool(Style style)
    : m_style(style), m_arena(1024)
{
    clear();
}

StringSlicePool::StringSlicePool(const ThisType& rhs)
    : m_style(rhs.m_style), m_arena(1024)
{
    // Set with rhs
    _set(rhs);
}

void StringSlicePool::operator=(const ThisType& rhs)
{
    if (&rhs != this)
    {
        _set(rhs);
    }
}

void StringSlicePool::_set(const ThisType& rhs)
{
    SLANG_ASSERT(this != &rhs);
    m_style = rhs.m_style;

    clear();

    const Index startIndex = rhs.getFirstAddedIndex();
    const Count count = rhs.m_slices.getCount();

    if (count == 0)
        return;

    // We need the same amount of slices
    m_slices.setCount(count);

    // Work out the total size to store all slices including terminating 0
    // (which *isn't* part of the slice size)
    size_t totalSize = 0;

    for (Index i = startIndex; i < count; ++i)
    {
        const auto slice = rhs.m_slices[i];
        totalSize += slice.getLength() + 1;
    }

    char* dst = (char*)m_arena.allocate(totalSize);

    for (Index i = startIndex; i < count; ++i)
    {
        const auto srcSlice = rhs.m_slices[i];
        const auto sliceSize = srcSlice.getLength();

        // Copy over the src slices contents
        ::memcpy(dst, srcSlice.begin(), sliceSize);
        // Zero terminate
        dst[sliceSize] = 0;

        const UnownedStringSlice dstSlice(dst, sliceSize);
        // Set the slice
        m_slices[i] = dstSlice;

        // Add to the map
        m_map.add(dstSlice, Handle(i));

        // Skip to next slices storage
        dst += sliceSize + 1;
    }
}

bool StringSlicePool::operator==(const ThisType& rhs) const
{
    if (this == &rhs)
    {
        return true;
    }

    if (m_style != rhs.m_style)
    {
        return false;
    }

    const auto count = m_slices.getCount();

    if (count != rhs.m_slices.getCount())
    {
        return false;
    }

    for (Index i = 0; i < count; ++i)
    {
        if (m_slices[i] != rhs.m_slices[i])
        {
            return false;
        }
    }

    return true;
}

void StringSlicePool::clear()
{
    m_map.clear();
    m_arena.deallocateAll();

    switch (m_style)
    {
    case Style::Default:
        {
            // Add the defaults
            m_slices.setCount(2);

            m_slices[0] = UnownedStringSlice((const char*)nullptr, (const char*)nullptr);
            m_slices[1] = UnownedStringSlice::fromLiteral("");

            // Add the empty entry
            m_map.add(m_slices[1], kEmptyHandle);
            break;
        }
    case Style::Empty:
        {
            // There are no defaults
            m_slices.clear();
            break;
        }
    }
}

void StringSlicePool::swapWith(ThisType& rhs)
{
    Swap(m_style, rhs.m_style);
    m_slices.swapWith(rhs.m_slices);
    m_map.swapWith(rhs.m_map);
    m_arena.swapWith(rhs.m_arena);
}

StringSlicePool::Handle StringSlicePool::add(const Slice& slice)
{
    const Handle* handlePtr = m_map.tryGetValue(slice);
    if (handlePtr)
    {
        return *handlePtr;
    }

    // Create a scoped copy
    UnownedStringSlice scopePath(
        m_arena.allocateString(slice.begin(), slice.getLength()),
        slice.getLength());

    const auto index = m_slices.getCount();

    m_slices.add(scopePath);
    m_map.add(scopePath, Handle(index));
    return Handle(index);
}

bool StringSlicePool::findOrAdd(const Slice& slice, Handle& outHandle)
{
    const Handle* handlePtr = m_map.tryGetValue(slice);
    if (handlePtr)
    {
        outHandle = *handlePtr;
        return true;
    }

    // Need to add.

    // Make a copy stored in the arena
    UnownedStringSlice scopeSlice(
        m_arena.allocateString(slice.begin(), slice.getLength()),
        slice.getLength());

    // Add using the arenas copy
    Handle newHandle = Handle(m_slices.getCount());
    m_map.add(scopeSlice, newHandle);

    // Add to slices list
    m_slices.add(scopeSlice);
    outHandle = newHandle;
    return false;
}

StringSlicePool::Handle StringSlicePool::add(StringRepresentation* stringRep)
{
    if (stringRep == nullptr && m_style == Style::Default)
    {
        return kNullHandle;
    }
    return add(StringRepresentation::asSlice(stringRep));
}

StringSlicePool::Handle StringSlicePool::add(const char* chars)
{
    switch (m_style)
    {
    case Style::Default:
        {
            if (!chars)
            {
                return kNullHandle;
            }
            if (chars[0] == 0)
            {
                return kEmptyHandle;
            }
            break;
        }
    case Style::Empty:
        {
            if (chars == nullptr)
            {
                SLANG_ASSERT(!"Empty style doesn't support nullptr");
                // Return an invalid handle
                return Handle(~HandleIntegral(0));
            }
        }
    }

    return add(UnownedStringSlice(chars));
}

Index StringSlicePool::findIndex(const Slice& slice) const
{
    const Handle* handlePtr = m_map.tryGetValue(slice);
    return handlePtr ? Index(*handlePtr) : -1;
}

ConstArrayView<UnownedStringSlice> StringSlicePool::getAdded() const
{
    const Index firstIndex = getFirstAddedIndex();
    return makeConstArrayView(m_slices.getBuffer() + firstIndex, m_slices.getCount() - firstIndex);
}

} // namespace Slang
