// slang-offset-container.h
#ifndef SLANG_OFFSET_CONTAINER_H_INCLUDED
#define SLANG_OFFSET_CONTAINER_H_INCLUDED

#include "slang-basic.h"

namespace Slang
{

/*
The purpose of OffsetContainer and related types is to provide a mechanism to easily serialize
offset structures.

The root idea here is the "offset pointer". A typical pointer in a language like C/C++ holds the
absolute address in the current address space of the thing that is being pointed to. This introduces
a problem, as when data is serialized in the contents will very likely be be placed at different
addresses - meaning any absolute pointer will point to the wrong place. There is also a related
issue around pointer sizes - on some targets they are 32 bits and on others 64 bits.

An offset pointer means a pointer that points to something 'offset' to some base address. The
OffsetPtr uses a 32 bit offset from the pointers location in memory. This means such a pointer can
address a 4Gb address space.

Special care is needed when using offset pointers - both when constructing structures that contain
them, reading them and in general usage.

For simplicity here we store all offset pointers within a single contiguous allocation. This
allocation is typically managed by the OffsetContainer for writing. When reading a MemoryOffsetBase
can be used.

An issue around using offset pointers, is that we cannot directly access it's contents, because it's
just an offset to some base address. Thus to access the thing being pointed to we need to turn the
offset pointer back into a 'raw' pointer. This is achieved via using the asRaw methods on the
OffsetBase. For a convenience operator[] can also be used, and this is typically the preferred
mechanism.

NOTE! That the evaluation order of a function calls parameters is undefined in C++. That whilst it
might appear doing

```
base[thing] = container.newObject<Thing>();
```

will evaluate the construction of newObject *before* the assignment, if you look at the assignment
as being a function call (as it is when it is overloaded), then base[thing] might be evaluated
*before* newObject, and if it is then the result could be wrong if the newObject needed to
reallocate. Therefore when allocation is involved, a new (or any allocation backed function call
from the OffsetContainer) should always place a result in a local variable. Then assign as in

```
auto anotherThing = container.newObject<Thing>();
base[thing] = anotherThing;
```

When creating structures - unless you know the allocated space (in the OffsetContainer or some other
piece of memory) is larger than required, then special care is needed, because when a new larger
piece of memory is allocated to hold everything, raw pointers pointers will likely be invalidated.
When reading there is typically no need to move the base address, so raw pointers remain valid
through out. When doing writing if a call is made to something that allocates memory on the
OffsetContainer - any raw pointer should be assumed invalid.

For example

```

struct Thing
{
    Offset32Ptr<OffsetString> text;
    int value;
};

void func()
{
    OffsetContainer container;
    OffsetBase& base = container.asBase();

    {
        // We can allocate on the heap. BUT we can't set up a offset pointer to it
        Thing thing;
        // BAD!! Will assert, because thing is not in the address range recorded in base.
        Offset32Ptr<Thing> thingOffsetPtr= base->asPtr(&thing);
    }

    // Ok - this is now correct
    Offset32Ptr<Thing> thing = container.newObject<Thing>();

    // To write values, we need a raw pointer
    {
        // To get the raw pointer we can use 'asRaw'
        auto rawThing = base->asRaw(thing);

        // Or more perhaps slightly more conveniently []
        auto rawThing = base[thing];

        // We can write and read things via the Safe32Ptr
        rawThing->value = 10;
        const int value = rawThing->value;

        SLANG_ASSERT(value == 10);
    }

    // Now lets write to it
    {
        // We can have raw pointer (or reference) to a thing but we need to be *careful* if we
allocate Thing* rawThing = base[thing];
        // We are okay here, nothing between getting the raw pointer and the write allocated/newed
anything on the OffsetContainer rawThing->value = 20;

        // Lets set up name
        Offset32Ptr<OffsetString> text = offsetContainer.newString("Hello World!");

        // BAD! The rawThing point could now be invalid because the call to newString may have had
to allocate more memory rawThing->text = text;

        // This is okay
        base[thing]->text = text;

        // Or we can update rawThing such that is up to date
        rawThing = base[thing];
        // So now this is okay again
        rawThing->text = text;

        // BAD! we don't know the evaluation order here, if the lhs is evaluate before the rhs, then
it could write to the wrong area of memory. base[thing]->text = offsetContainer.newString("Hello
World again!");

        // So where there is allocation, and assignment to something that in held in offset ptr use
a local for the allocation as in
        {
            auto text = offsetContainer.newString("Hello World again!");
            base[thing]->text = text;
        }
    }
}

```
*/

enum
{
    kNull32Offset = 0,
    kStartOffset = uint32_t(sizeof(uint64_t)), ///< The offset to the first contained thing
};

template<typename T>
class Offset32Ref;

/* A pointer to items held in OffsetContainer (or OffsetBase relative) that remains correct even if
the memory inside OffsetContainer moves.
*/
template<typename T>
class Offset32Ptr
{
public:
    typedef Offset32Ptr ThisType;

    const ThisType& operator=(const ThisType& rhs)
    {
        m_offset = rhs.m_offset;
        return *this;
    }
    bool operator==(const ThisType& rhs) const { return m_offset == rhs.m_offset; }
    bool operator!=(const ThisType& rhs) const { return m_offset != rhs.m_offset; }

    bool operator<(const ThisType& rhs) const { return m_offset < rhs.m_offset; }
    bool operator<=(const ThisType& rhs) const { return m_offset <= rhs.m_offset; }
    bool operator>(const ThisType& rhs) const { return m_offset > rhs.m_offset; }
    bool operator>=(const ThisType& rhs) const { return m_offset >= rhs.m_offset; }

    operator bool() const { return m_offset != kNull32Offset; }

    Offset32Ref<T> operator*();

    ThisType& operator++()
    {
        m_offset += uint32_t(sizeof(T));
        return *this;
    }
    ThisType operator++(int)
    {
        const auto offset = m_offset;
        m_offset += uint32_t(sizeof(T));
        return ThisType(offset);
    }

    ThisType& operator--()
    {
        m_offset -= sizeof(T);
        return *this;
    }
    ThisType operator--(int)
    {
        const auto offset = m_offset;
        m_offset -= uint32_t(sizeof(T));
        return ThisType(offset);
    }

    friend ThisType operator+(const ThisType& a, Index b)
    {
        return ThisType(a.m_offset + uint32_t(sizeof(T) * b));
    }
    friend ThisType operator+(Index a, const ThisType& b)
    {
        return ThisType(b.m_offset + uint32_t(sizeof(T) * a));
    }

    bool isNull() const { return m_offset == kNull32Offset; }

    void setNull() { m_offset = kNull32Offset; }
    Offset32Ptr()
        : m_offset(kNull32Offset)
    {
    }
    Offset32Ptr(const ThisType& rhs)
        : m_offset(rhs.m_offset)
    {
    }
    explicit Offset32Ptr(uint32_t offset)
        : m_offset(offset)
    {
    }

    uint32_t m_offset;
};

/* A reference to items held in OffsetContainer (or OffsetBase relative) that remains correct even
if the memory inside OffsetContainer moves.
*/
template<typename T>
class Offset32Ref
{
public:
    typedef Offset32Ref ThisType;

    const ThisType& operator=(const ThisType& rhs)
    {
        m_offset = rhs.m_offset;
        return *this;
    }

    Offset32Ptr<T> operator&() { return Offset32Ptr<T>(m_offset); }

    Offset32Ref(const ThisType& rhs)
        : m_offset(rhs.m_offset)
    {
    }
    explicit Offset32Ref(uint32_t offset)
        : m_offset(offset)
    {
        SLANG_ASSERT(offset != kNull32Offset);
    }

    uint32_t m_offset;
};

// ---------------------------------------------------------------------------
template<typename T>
SLANG_FORCE_INLINE Offset32Ref<T> Offset32Ptr<T>::operator*()
{
    return Offset32Ref<T>(m_offset);
}


/* Much like Offset32Ptr this is an array but whose memory is stored inside the OffsetContainer.
This means elements types must be 'offset types'. */
template<typename T>
class Offset32Array
{
public:
    Offset32Ptr<const T> begin() const { return Offset32Ptr<const T>(m_data.m_offset); }
    Offset32Ptr<const T> end() const { return begin() + Index(m_count); }

    Offset32Ptr<T> begin() { return m_data; }
    Offset32Ptr<T> end() { return begin() + Index(m_count); }

    Index getCount() const { return Index(m_count); }

    Offset32Ref<const T> operator[](Index i) const
    {
        SLANG_ASSERT(i >= 0 && uint32_t(i) < m_count);
        return Offset32Ref<const T>((m_data + i).m_offset);
    }
    Offset32Ref<T> operator[](Index i)
    {
        SLANG_ASSERT(i >= 0 && uint32_t(i) < m_count);
        return Offset32Ref<T>((m_data + i).m_offset);
    }

    Offset32Array(Offset32Ptr<T> data, uint32_t count)
        : m_data(data), m_count(count)
    {
    }

    Offset32Array()
        : m_count(0)
    {
    }

    Offset32Ptr<T> m_data;
    uint32_t m_count;
};

/** OffsetString is used for storing strings within a OffsetContainer. Strings are stored with the
initial byte indicating the size of the string. Note that all offset strings are stored with a
terminating zero, and that the terminating zero is *NOT* included in the encoded size. */
struct OffsetString
{
    enum
    {
        kSizeBase = 251,
        kMaxSizeEncodeSize = 5,
    };

    /// Get contents as a slice
    UnownedStringSlice getSlice() const;
    /// Get null terminated string
    const char* getCstr() const;

    /// Decode the size. Returns the start of the string text, and outSize holds the size (NOT
    /// including terminating 0)
    static const char* decodeSize(const char* in, size_t& outSize);

    /// Returns the amount of bytes used, end encoding in 'encode'
    static size_t calcEncodedSize(size_t size, uint8_t encode[kMaxSizeEncodeSize]);
    /// Calculate the total size needed to store the string *including* terminating 0
    static size_t calcAllocationSize(const UnownedStringSlice& slice);

    /// Calculate the total size needed to store string. Size should be passed *without* terminating
    /// 0
    static size_t calcAllocationSize(size_t size);

    char m_sizeThenContents[1];
};

/* A type that is used to hold the base address of the contiguous memory that holds either
 * Offset32Ptr and related types>
 */
class OffsetBase
{
public:
    typedef OffsetBase ThisType;

    /// Turn an offset into a raw regular pointer or reference
    template<typename T>
    T* asRaw(const Offset32Ptr<T>& ptr)
    {
        return (T*)_getRaw(ptr.m_offset);
    }
    template<typename T>
    T& asRaw(const Offset32Ref<T>& ref)
    {
        return *(T*)_getRaw(ref.m_offset);
    }

    /// A more terse way to get a raw pointer/reference. Using the [] operator can be seen as
    /// 'indexing' to access the object the offset relates to. Unlike 'indices' that are typically
    /// used with [] offsets are generally not contiguous.
    template<typename T>
    T* operator[](const Offset32Ptr<T>& ptr)
    {
        return (T*)_getRaw(ptr.m_offset);
    }
    template<typename T>
    T& operator[](const Offset32Ref<T>& ref)
    {
        return *(T*)_getRaw(ref.m_offset);
    }

    template<typename T>
    Offset32Ptr<T> asPtr(T* ptr)
    {
        return Offset32Ptr<T>(getOffset(ptr));
    }
    /// Note the use of ptr when setting up a reference here - it's needed because a ref does not
    /// have to be backed by a pointer. And commonly is not when the const& and the thing referenced
    /// can be held in a word.
    template<typename T>
    Offset32Ref<T> asRef(T* ptr)
    {
        SLANG_ASSERT(ptr);
        return Offset32Ref<T>(getOffset(ptr));
    }

    uint32_t getOffset(const void* ptr)
    {
        if (ptr == nullptr)
        {
            return kNull32Offset;
        }
        ptrdiff_t diff = ((const uint8_t*)ptr) - m_data;
        SLANG_ASSERT(diff > 0 && size_t(diff) < m_dataSize);
        return uint32_t(diff);
    }

    /// Get the contained data
    SLANG_FORCE_INLINE uint8_t* getData() { return m_data; }
    /// Return the last used byte of the data
    SLANG_FORCE_INLINE size_t getDataCount() const { return m_dataSize; }

    /// Get the first allocated thing. Typically the root of the structure contained
    void* getFirst() { return (m_dataSize < kStartOffset) ? nullptr : (m_data + kStartOffset); }

    /// Get a raw pointer from the offset
    uint8_t* _getRaw(uint32_t offset)
    {
        return (offset == kNull32Offset) ? nullptr : (m_data + offset);
    }

    OffsetBase()
        : m_data(nullptr), m_dataSize(0)
    {
    }


    uint8_t* m_data;
    size_t m_dataSize;

protected:
    /// We want protected, because we don't want copies to be made of OffsetBase by default!
    OffsetBase(const ThisType& rhs) = default;
    ThisType& operator=(const ThisType& rhs) = default;
};

class MemoryOffsetBase : public OffsetBase
{
public:
    void set(void* data, size_t dataSize)
    {
        m_data = (uint8_t*)data;
        m_dataSize = dataSize;
    }
};

/* OffsetContainer is a type designed to manage the construction structures around 'offset types'.
In particular it allows for construction of offset structures where their total encoded size is not
known at the outset.

The main mechanism to make this work is via the use of OffsetXXX types, which when constructed from
the OffsetContainer will maintain valid values, even if the underlying backing memories location is
changed.
*/
class OffsetContainer : public OffsetBase
{
public:
    template<typename T>
    Offset32Ptr<T> newObject()
    {
        void* data = allocate(sizeof(T), SLANG_ALIGN_OF(T));
        new (data) T();
        return Offset32Ptr<T>(getOffset(data));
    }

    template<typename T>
    Offset32Array<T> newArray(size_t size)
    {
        if (size == 0)
        {
            return Offset32Array<T>();
        }
        T* data = (T*)allocate(sizeof(T) * size, SLANG_ALIGN_OF(T));
        for (size_t i = 0; i < size; ++i)
        {
            new (data + i) T();
        }
        return Offset32Array<T>(Offset32Ptr<T>(getOffset(data)), uint32_t(size));
    }

    /// Get the base - which is needed for turning offsets into things
    OffsetBase& asBase() { return *this; }

    /// Allocate without alignment (effectively 1)
    void* allocate(size_t size);
    void* allocate(size_t size, size_t alignment);
    void* allocateAndZero(size_t size, size_t alignment);

    void fixAlignment(size_t alignment);

    Offset32Ptr<OffsetString> newString(const UnownedStringSlice& slice);
    Offset32Ptr<OffsetString> newString(const char* contents);

    /// Ctor
    OffsetContainer();
    ~OffsetContainer();

protected:
    size_t m_capacity;
};

} // namespace Slang

#endif
