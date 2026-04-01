#ifndef SLANG_CORE_STRING_SLICE_POOL_H
#define SLANG_CORE_STRING_SLICE_POOL_H

#include "slang-array-view.h"
#include "slang-dictionary.h"
#include "slang-list.h"
#include "slang-memory-arena.h"
#include "slang-string.h"

namespace Slang
{

/* Holds a unique set of slices.

Note that all slices (except kNullHandle) are stored with terminating zeros.

The default handles kNullHandle, kEmptyHandle can only be used on a StringSlicePool
initialized with the Style::Default. Not doing so will return an undefined result.

TODO(JS):
An argument could be made to make different classes, perhaps deriving from a base class
that exhibited the two behaviors. That doing so would make the default handles defined
for that class for example.

This is a little awkward in practice, because behavior of some methods need to change
(like adding a c string with nullptr, or clearing, as well as some other perhaps less necessary
optimizations). This could be achieved via virtual functions, but this all seems overkill.
*/
class StringSlicePool
{
public:
    typedef StringSlicePool ThisType;
    typedef uint32_t HandleIntegral;

    enum class Style
    {
        Default, ///< Default style - has default handles (like kNullHandle and kEmptyHandle)
        Empty,   ///< Empty style - has no handles by default. Using default handles will likely
                 ///< produce the wrong result.
    };

    enum class Handle : HandleIntegral;
    typedef UnownedStringSlice Slice;

    /// The following default handles *only* apply if constructed with the Style::Default

    /// Handle of 0 is null. If accessed will be returned as the empty string with nullptr the chars
    static const Handle kNullHandle = Handle(0);
    /// Handle of 1 is the empty string.
    static const Handle kEmptyHandle = Handle(1);

    static const Index kDefaultHandlesCount = 2;

    /// Returns the index of a slice, if contained, or -1 if not found
    Index findIndex(const Slice& slice) const;

    /// True if has the slice
    bool has(const Slice& slice) { return findIndex(slice) >= 0; }
    /// Add a slice
    Handle add(const Slice& slice);
    /// Add from a string
    Handle add(const char* chars);
    /// Add a StringRepresentation
    Handle add(StringRepresentation* string);
    /// Add a string
    Handle add(const String& string) { return add(string.getUnownedSlice()); }

    /// Add and get the result as a slice
    Slice addAndGetSlice(const Slice& slice) { return getSlice(add(slice)); }
    Slice addAndGetSlice(const char* chars) { return getSlice(add(chars)); }
    Slice addAndGetSlice(const String& string) { return getSlice(add(string)); }

    /// Returns true if found
    bool findOrAdd(const Slice& slice, Handle& outHandle);

    /// Empty contents
    void clear();

    /// Get the slice from the handle
    const UnownedStringSlice& getSlice(Handle handle) const { return m_slices[UInt(handle)]; }

    /// Get all the slices
    const List<UnownedStringSlice>& getSlices() const { return m_slices; }

    /// Get the number of slices
    Index getSlicesCount() const { return m_slices.getCount(); }

    /// Returns true if the handle is a default one. Only meaningful on a Style::Default.
    bool isDefaultHandle(Handle handle) const
    {
        SLANG_ASSERT(
            m_style == Style::Default &&
            // TODO(C++20), use bit_cast here
            HandleIntegral(handle) <= HandleIntegral(std::numeric_limits<Index>::max()));
        return Index(handle) < kDefaultHandlesCount;
    }

    /// Convert a handle to and index. (A handle is just an index!)
    static Index asIndex(Handle handle) { return Index(handle); }

    /// Get the style of the pool
    Style getStyle() const { return m_style; }

    /// Get all the added slices (does not have default slices, if there are any)
    ConstArrayView<UnownedStringSlice> getAdded() const;

    /// Get the index of the first added handle
    Index getFirstAddedIndex() const
    {
        return m_style == Style::Default ? kDefaultHandlesCount : 0;
    }

    /// Swap this with rhs
    void swapWith(ThisType& rhs);

    /// True if the pools are identical. Same style, same slices in the same order.
    bool operator==(const ThisType& rhs) const;

    /// Copy ctor
    StringSlicePool(const ThisType& rhs);
    /// Assignment
    void operator=(const ThisType& rhs);

    /// Ctor
    explicit StringSlicePool(Style style);

protected:
    void _set(const ThisType& rhs);

    Style m_style;
    List<UnownedStringSlice> m_slices;
    Dictionary<UnownedStringSlice, Handle> m_map;
    MemoryArena m_arena;
};

} // namespace Slang

#endif // SLANG_STRING_SLICE_POOL_H
