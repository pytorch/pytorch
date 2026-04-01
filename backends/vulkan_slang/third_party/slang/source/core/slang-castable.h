// slang-castable.h
#ifndef SLANG_CASTABLE_H
#define SLANG_CASTABLE_H


#include "../core/slang-com-object.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"

namespace Slang
{

// Dynamic cast of ICastable derived types
template<typename T>
SLANG_FORCE_INLINE T* dynamicCast(ICastable* castable)
{
    if (castable)
    {
        void* obj = castable->castAs(T::getTypeGuid());
        return obj ? reinterpret_cast<T*>(obj) : ((T*)nullptr);
    }
    return nullptr;
}

// as style cast
template<typename T>
SLANG_FORCE_INLINE T* as(ICastable* castable)
{
    if (castable)
    {
        void* obj = castable->castAs(T::getTypeGuid());
        return obj ? reinterpret_cast<T*>(obj) : ((T*)nullptr);
    }
    return nullptr;
}

/* An interface for boxing values */
class IBoxValueBase : public ICastable
{
    SLANG_COM_INTERFACE(
        0x8b4aad81,
        0x4934,
        0x4a67,
        {0xb2, 0xe2, 0xe9, 0x17, 0xfc, 0x29, 0x12, 0x54});

    /// Get the contained object
    virtual SLANG_NO_THROW void* SLANG_MCALL getValuePtr() = 0;
    /// Get the guid that represents the contained ref object
    virtual SLANG_NO_THROW SlangUUID SLANG_MCALL getValueTypeGuid() = 0;
};

template<typename T>
class IBoxValue : public IBoxValueBase
{
public:
    SLANG_FORCE_INLINE T& get() { return *reinterpret_cast<T*>(getValuePtr()); }
    SLANG_FORCE_INLINE T* getPtr() { return reinterpret_cast<T*>(getValuePtr()); }
};

// Cast into a boxed value type
template<typename T>
SLANG_FORCE_INLINE IBoxValue<T>* asBoxValue(ICastable* castable)
{
    IBoxValueBase* base = as<IBoxValueBase>(castable);
    return (base && base->getValueTypeGuid() == T::getTypeGuid()) ? static_cast<IBoxValue<T>*>(base)
                                                                  : nullptr;
}

template<typename T>
class BoxValue : public ComBaseObject, public IBoxValue<T>
{
public:
    SLANG_COM_BASE_IUNKNOWN_ALL

    // ICastable
    SLANG_NO_THROW void* SLANG_MCALL castAs(const Guid& guid) SLANG_OVERRIDE;

    // IBoxValue
    virtual SLANG_NO_THROW void* SLANG_MCALL getValuePtr() SLANG_OVERRIDE { return &m_value; }
    virtual SLANG_NO_THROW SlangUUID SLANG_MCALL getValueTypeGuid() SLANG_OVERRIDE
    {
        return T::getTypeGuid();
    }

    BoxValue() {}

    explicit BoxValue(const T& rhs)
        : m_value(rhs)
    {
    }

protected:
    void* getInterface(const Guid& guid);
    void* getObject(const Guid& guid);

    T m_value;
};

// ------------------------------------------------------------
template<typename T>
void* BoxValue<T>::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ICastable::getTypeGuid() ||
        guid == IBoxValueBase::getTypeGuid())
    {
        return static_cast<IBoxValueBase*>(this);
    }
    return nullptr;
}

// ------------------------------------------------------------
template<typename T>
void* BoxValue<T>::getObject(const Guid& guid)
{
    if (guid == T::getTypeGuid())
    {
        return &m_value;
    }
    return nullptr;
}

// ------------------------------------------------------------
template<typename T>
void* BoxValue<T>::castAs(const Guid& guid)
{
    if (auto ptr = getObject(guid))
    {
        return ptr;
    }
    return getInterface(guid);
}

/* Adapter interface to make a non castable types work as ICastable */
class IUnknownCastableAdapter : public ICastable
{
    SLANG_COM_INTERFACE(
        0x8b4aad81,
        0x4934,
        0x4a67,
        {0xb2, 0xe2, 0xe9, 0x17, 0xfc, 0x29, 0x12, 0x54});

    /// When using the adapter, this provides a way to directly get the internal no ICastable type
    virtual SLANG_NO_THROW ISlangUnknown* SLANG_MCALL getContained() = 0;
};


/* An adapter such that types which aren't derived from ICastable, can be used as such.

With the following caveats.
* the interfaces/objects of the adapter are checked *first*, so IUnknown will always be for the
adapter
* assumes when doing a queryInterface on the contained item, it will remain in scope when released
(this is *not* strict COM)
*/
class UnknownCastableAdapter : public ComBaseObject, public IUnknownCastableAdapter
{
public:
    SLANG_COM_BASE_IUNKNOWN_ALL

    // ICastable
    SLANG_NO_THROW void* SLANG_MCALL castAs(const Guid& guid) SLANG_OVERRIDE;

    // IUnknownCastableAdapter
    virtual SLANG_NO_THROW ISlangUnknown* SLANG_MCALL getContained() SLANG_OVERRIDE
    {
        return m_contained;
    }

    UnknownCastableAdapter(ISlangUnknown* unk)
        : m_contained(unk)
    {
        SLANG_ASSERT(unk);
    }

protected:
    void* getInterface(const Guid& guid);
    void* getObject(const Guid& guid);

    ComPtr<ISlangUnknown> m_contained;

    // We hold a cache for a single lookup to make things a little faster
    void* m_found = nullptr;
    Guid m_foundGuid;
};

struct CastableUtil
{
    /// Given an ISlangUnkown return as a castable interface.
    /// Can use UnknownCastableAdapter if can't queryInterface unk to ICastable
    static ComPtr<ICastable> getCastable(ISlangUnknown* unk);
};


// A way to clone an interface (that derives from IClonable) such that it returns an interface
// of the same type.
template<typename T>
SLANG_FORCE_INLINE ComPtr<T> cloneInterface(T* in)
{
    SLANG_ASSERT(in);
    // Must be derivable from clonable
    IClonable* clonable = in;
    // We can clone with the same interface
    T* clone = (T*)clonable->clone(T::getTypeGuid());
    // Clone must exist
    SLANG_ASSERT(clone);
    return ComPtr<T>(clone);
}

} // namespace Slang

#endif
