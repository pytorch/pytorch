#ifndef SLANG_CORE_BLOB_H
#define SLANG_CORE_BLOB_H

#include "../core/slang-com-object.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang-list.h"
#include "slang-string.h"
#include "slang.h"

#include <stdarg.h>

namespace Slang
{

/** Base class for simple blobs.
 */
class BlobBase : public ISlangBlob, public ICastable, public ComBaseObject
{
public:
    // ISlangUnknown
    SLANG_COM_BASE_IUNKNOWN_ALL

    // ICastable
    virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const SlangUUID& guid) SLANG_OVERRIDE;

protected:
    ISlangUnknown* getInterface(const Guid& guid);
    void* getObject(const Guid& guid);
};

/** A blob that uses a `StringRepresentation` for its storage.

By design the StringBlob owns a unique reference to the StringRepresentation.
This is because StringBlob, implements an interface which should work across threads.
*/
class StringBlob : public BlobBase
{
public:
    SLANG_CLASS_GUID(0xf7e0e93c, 0xde70, 0x4531, {0x9c, 0x9f, 0xdd, 0xa3, 0xf6, 0xc6, 0xc0, 0xdd});

    // ICastable
    virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const SlangUUID& guid) SLANG_OVERRIDE;

    // ISlangBlob
    SLANG_NO_THROW void const* SLANG_MCALL getBufferPointer() SLANG_OVERRIDE
    {
        return m_slice.begin();
    }
    SLANG_NO_THROW size_t SLANG_MCALL getBufferSize() SLANG_OVERRIDE { return m_slice.getLength(); }

    /// Since in is not being moved will *always* create a new representation, unless the in is
    /// empty
    static ComPtr<ISlangBlob> create(const String& in);
    /// Create from a slice
    static ComPtr<ISlangBlob> create(const UnownedStringSlice& slice);

    /// Moves from in into the created blob.
    /// NOTE! That will only use the representation from in, if it is *unique*
    /// otherwise it will make a new copy.
    static ComPtr<ISlangBlob> moveCreate(String& in);
    static ComPtr<ISlangBlob> moveCreate(String&& in);

    /// Dtor
    ~StringBlob();

protected:
    /// Init with a rep when can't be owned.
    void _setWithCopy(StringRepresentation* rep);
    /// Init with a representation that has been moved.
    void _setWithMove(StringRepresentation* rep);

    /// Create a unique copy of rep.
    /// If nullptr will work (if rep is empty, will return that)
    static StringRepresentation* _createUniqueCopy(StringRepresentation* rep);

    /// Rep can only be nullptr or have a single ref
    void _setUniqueRep(StringRepresentation* rep);

    void* getObject(const Guid& guid);

    UnownedTerminatedStringSlice m_slice; ///< The contents
    StringRepresentation* m_uniqueRep =
        nullptr; ///< Holds actual bytes. Can be nullptr if it's an empty string.
};

class ListBlob : public BlobBase
{
public:
    typedef BlobBase Super;
    typedef ListBlob ThisType;

    // ICastable
    virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const SlangUUID& guid) SLANG_OVERRIDE;
    // ISlangBlob
    SLANG_NO_THROW void const* SLANG_MCALL getBufferPointer() SLANG_OVERRIDE
    {
        return m_data.getBuffer();
    }
    SLANG_NO_THROW size_t SLANG_MCALL getBufferSize() SLANG_OVERRIDE { return m_data.getCount(); }

    static ComPtr<ISlangBlob> create(const List<uint8_t>& data)
    {
        return ComPtr<ISlangBlob>(new ListBlob(data));
    }

    static ComPtr<ISlangBlob> moveCreate(List<uint8_t>& data)
    {
        return ComPtr<ISlangBlob>(new ListBlob(_Move(data)));
    }

protected:
    explicit ListBlob(const List<uint8_t>& data)
        : m_data(data)
    {
    }
    // Move ctor
    explicit ListBlob(List<uint8_t>&& data)
        : m_data(data)
    {
    }

    void* getObject(const Guid& guid);

    void operator=(const ThisType& rhs) = delete;

    List<uint8_t> m_data;
};

class ScopedAllocation
{
public:
    typedef ScopedAllocation ThisType;
    // Returns the allocation if successful.
    void* allocate(size_t size)
    {
        deallocate();
        if (size > 0)
        {
            m_data = ::malloc(size);
        }
        m_sizeInBytes = size;
        m_capacityInBytes = size;
        return m_data;
    }
    /// Allocate size including a 0 byte at `size`.
    void* allocateTerminated(size_t size)
    {
        SLANG_ASSUME(size != std::numeric_limits<size_t>::max());
        uint8_t* data = (uint8_t*)allocate(size + 1);
        data[size] = 0;
        m_sizeInBytes = size;
        return data;
    }
    /// Deallocates if holds an allocation
    void deallocate()
    {
        if (m_data)
        {
            ::free(m_data);
            m_data = nullptr;
        }
        m_sizeInBytes = 0;
        m_capacityInBytes = 0;
    }
    // Reallocate so the buffer is the specified capacity/size. Contents of buffer up to size remain
    // intact.
    void reallocate(size_t capacity)
    {
        if (capacity != m_capacityInBytes)
        {
            m_data = ::realloc(m_data, capacity);
            m_sizeInBytes = capacity;
            m_capacityInBytes = capacity;
        }
    }
    /// Makes this no longer own the allocation. Returns the allocated data (or nullptr if no
    /// allocation)
    void* detach()
    {
        void* data = m_data;
        m_data = nullptr;
        m_sizeInBytes = 0;
        m_capacityInBytes = 0;
        return data;
    }
    /// Attach some data.
    /// NOTE! data must be a pointer that was returned from malloc, otherwise will incorrectly free.
    void attach(void* data, size_t size)
    {
        deallocate();
        m_data = data;
        m_sizeInBytes = size;
        m_capacityInBytes = size;
    }

    void* set(const void* data, size_t size)
    {
        void* dst = allocate(size);
        if (dst)
        {
            memcpy(dst, data, size);
        }
        return dst;
    }

    /// Get the allocated data. Returns nullptr if there is no allocated data
    void* getData() const { return m_data; }
    /// Get the size of the allocated data.
    size_t getSizeInBytes() const { return m_sizeInBytes; }
    /// Get the capacity in bytes
    size_t getCapacityInBytes() const { return m_capacityInBytes; }

    void setSizeInBytes(size_t size)
    {
        SLANG_ASSERT(size <= m_capacityInBytes);
        m_sizeInBytes = size;
    }

    void swap(ThisType& rhs)
    {
        Swap(m_data, rhs.m_data);
        Swap(m_sizeInBytes, rhs.m_sizeInBytes);
        Swap(m_capacityInBytes, rhs.m_capacityInBytes);
    }

    /// True if has zero termination, at the byte at m_sizeInBytes
    bool isTerminated() const
    {
        return m_capacityInBytes > m_sizeInBytes && ((const char*)m_data)[m_sizeInBytes] == 0;
    }

    ScopedAllocation()
        : m_data(nullptr), m_sizeInBytes(0), m_capacityInBytes(0)
    {
    }

    ~ScopedAllocation() { deallocate(); }

private:
    // disable
    ScopedAllocation(const ThisType& rhs) = delete;
    void operator=(const ThisType& rhs) = delete;

    void* m_data;
    size_t m_sizeInBytes;
    size_t m_capacityInBytes;
};

/** A blob that manages some raw data that it owns.
 */
class RawBlob : public BlobBase
{
public:
    // ICastable
    virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const SlangUUID& guid) SLANG_OVERRIDE;
    // ISlangBlob
    SLANG_NO_THROW void const* SLANG_MCALL getBufferPointer() SLANG_OVERRIDE
    {
        return m_data.getData();
    }
    SLANG_NO_THROW size_t SLANG_MCALL getBufferSize() SLANG_OVERRIDE
    {
        return m_data.getSizeInBytes();
    }

    static ComPtr<ISlangBlob> moveCreate(ScopedAllocation& alloc)
    {
        RawBlob* blob = new RawBlob;
        blob->m_data.swap(alloc);
        return ComPtr<ISlangBlob>(blob);
    }

    /// Create a blob that will retain (a copy of) raw data.
    static inline ComPtr<ISlangBlob> create(void const* inData, size_t size)
    {
        return ComPtr<ISlangBlob>(new RawBlob(inData, size));
    }

protected:
    // Ctor
    // NOTE! Takes a copy of the input data
    RawBlob(const void* data, size_t size) { memcpy(m_data.allocateTerminated(size), data, size); }

    void* getObject(const Guid& guid);

    RawBlob() = default;

    ScopedAllocation m_data;
};

// A blob that does not own it's contained data.
class UnownedRawBlob : public BlobBase
{
public:
    // ISlangBlob
    SLANG_NO_THROW void const* SLANG_MCALL getBufferPointer() SLANG_OVERRIDE { return m_data; }
    SLANG_NO_THROW size_t SLANG_MCALL getBufferSize() SLANG_OVERRIDE { return m_dataSizeInBytes; }

    static inline ComPtr<ISlangBlob> create(void const* inData, size_t size)
    {
        return ComPtr<ISlangBlob>(new UnownedRawBlob(inData, size));
    }

protected:
    // Ctor
    UnownedRawBlob(const void* data, size_t size)
        : m_data(data), m_dataSizeInBytes(size)
    {
    }

    UnownedRawBlob() = default;

    const void* m_data;
    size_t m_dataSizeInBytes;
};

/** A Blob that has no ref counting and exists typically for entire execution.
The memory it references is *not* owned by the blob.
This is useful when a Blob is useful to represent some global immutable chunk of memory.
*/
class StaticBlob : public ISlangBlob, public ICastable
{
public:
    // ISlangUnknown
    SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(SlangUUID const& uuid, void** outObject)
        SLANG_OVERRIDE;
    SLANG_NO_THROW uint32_t SLANG_MCALL addRef() SLANG_OVERRIDE { return 1; }
    SLANG_NO_THROW uint32_t SLANG_MCALL release() SLANG_OVERRIDE { return 1; }

    // ICastable
    virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const SlangUUID& guid) SLANG_OVERRIDE;

    // ISlangBlob
    SLANG_NO_THROW void const* SLANG_MCALL getBufferPointer() SLANG_OVERRIDE { return m_data; }
    SLANG_NO_THROW size_t SLANG_MCALL getBufferSize() SLANG_OVERRIDE { return m_dataCount; }

    StaticBlob(const void* data, size_t dataCount)
        : m_data(data), m_dataCount(dataCount)
    {
    }

protected:
    ISlangUnknown* getInterface(const Guid& guid);
    void* getObject(const Guid& guid);

    const void* m_data;
    size_t m_dataCount;
};

class ScopeBlob : public BlobBase
{
public:
    // ICastable
    virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const SlangUUID& guid) SLANG_OVERRIDE;

    // ISlangBlob
    SLANG_NO_THROW void const* SLANG_MCALL getBufferPointer() SLANG_OVERRIDE
    {
        return m_blob->getBufferPointer();
    }
    SLANG_NO_THROW size_t SLANG_MCALL getBufferSize() SLANG_OVERRIDE
    {
        return m_blob->getBufferSize();
    }

    static inline ComPtr<ISlangBlob> create(ISlangBlob* blob, ISlangUnknown* scope)
    {
        return ComPtr<ISlangBlob>(new ScopeBlob(blob, scope));
    }

protected:
    // Ctor
    ScopeBlob(ISlangBlob* blob, ISlangUnknown* scope)
        : m_blob(blob), m_scope(scope)
    {
        // Cache the ICastable interface if there is one.
        blob->queryInterface(SLANG_IID_PPV_ARGS(m_castable.writeRef()));
    }

    ComPtr<ISlangUnknown> m_scope;
    ComPtr<ISlangBlob> m_blob;
    ComPtr<ICastable>
        m_castable; ///< Set if the blob has this interface. Set to nullptr if does not.
};

} // namespace Slang

#endif // SLANG_CORE_BLOB_H
