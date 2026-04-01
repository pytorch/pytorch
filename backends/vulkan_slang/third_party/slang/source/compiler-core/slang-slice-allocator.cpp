// slang-slice-allocator.cpp
#include "slang-slice-allocator.h"

#include "../core/slang-blob.h"

namespace Slang
{

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SliceUtil !!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* static */ List<String> SliceUtil::toList(const Slice<TerminatedCharSlice>& in)
{
    List<String> list;
    const auto count = in.count;

    list.setCount(count);
    for (Index i = 0; i < count; ++i)
    {
        list[i] = asStringSlice(in[i]);
    }
    return list;
}

/* static */ const char* SliceUtil::getTerminated(ISlangBlob* blob, TerminatedCharSlice& outSlice)
{
    const auto size = blob->getBufferSize();
    if (size == 0)
    {
        outSlice = TerminatedCharSlice();
        return outSlice.begin();
    }

    // If there is a 0 at the end byte, we are zero terminated
    const char* chars = (const char*)blob->getBufferPointer();
    if (chars[size - 1] == 0)
    {
        outSlice = TerminatedCharSlice(chars, Count(size - 1));
        return chars;
    }

    // See if it has a castable interface
    ComPtr<ICastable> castable;
    if (SLANG_SUCCEEDED(blob->queryInterface(SLANG_IID_PPV_ARGS(castable.writeRef()))))
    {
        if (castable->castAs(SlangTerminatedChars::getTypeGuid()))
        {
            outSlice = TerminatedCharSlice(chars, Count(size));
            return chars;
        }
    }

    return nullptr;
}

/* static */ TerminatedCharSlice SliceUtil::toTerminatedCharSlice(
    SliceAllocator& allocator,
    ISlangBlob* blob)
{
    TerminatedCharSlice slice;
    if (SliceUtil::getTerminated(blob, slice))
    {
        return slice;
    }
    const auto size = blob->getBufferSize();
    // We are out of options, we just have to allocate with zero termination which allocateString
    // does
    auto dst =
        allocator.getArena().allocateString((const char*)blob->getBufferPointer(), Count(size));
    return TerminatedCharSlice(dst, Count(size));
}

/* static */ TerminatedCharSlice SliceUtil::toTerminatedCharSlice(
    StringBuilder& storage,
    ISlangBlob* blob)
{
    TerminatedCharSlice slice;
    if (SliceUtil::getTerminated(blob, slice))
    {
        return slice;
    }

    const auto size = blob->getBufferSize();
    auto chars = (const char*)blob->getBufferPointer();

    storage.clear();
    storage.append(UnownedStringSlice(chars, size));

    return TerminatedCharSlice(storage.getBuffer(), Count(size));
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SliceAllocator !!!!!!!!!!!!!!!!!!!!!!!!!!! */

TerminatedCharSlice SliceAllocator::allocate(const char* in)
{
    const size_t length = ::strlen(in);
    auto dst = m_arena.allocateString(in, length);
    return TerminatedCharSlice(dst, length);
}

TerminatedCharSlice SliceAllocator::allocate(const UnownedStringSlice& slice)
{
    const auto length = slice.getLength();
    auto dst = m_arena.allocateString(slice.begin(), length);
    return TerminatedCharSlice(dst, length);
}

TerminatedCharSlice SliceAllocator::allocate(const Slice<char>& slice)
{
    const auto count = slice.count;
    auto dst = m_arena.allocateString(slice.begin(), count);
    return TerminatedCharSlice(dst, count);
}

Slice<TerminatedCharSlice> SliceAllocator::allocate(const List<String>& in)
{
    const auto count = in.getCount();
    if (count == 0)
    {
        return Slice<TerminatedCharSlice>(nullptr, 0);
    }

    auto dst = m_arena.allocateArray<TerminatedCharSlice>(count);
    for (Index i = 0; i < count; ++i)
    {
        dst[i] = allocate(in[i]);
    }

    return Slice<TerminatedCharSlice>(dst, count);
}

} // namespace Slang
