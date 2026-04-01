#include "slang-blob.h"

namespace Slang
{

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! BlobBase !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

ISlangUnknown* BlobBase::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ISlangBlob::getTypeGuid())
    {
        return static_cast<ISlangBlob*>(this);
    }
    if (guid == ICastable::getTypeGuid())
    {
        return static_cast<ICastable*>(this);
    }
    return nullptr;
}

void* BlobBase::getObject(const Guid& guid)
{
    SLANG_UNUSED(guid);
    return nullptr;
}

void* BlobBase::castAs(const SlangUUID& guid)
{
    if (auto intf = getInterface(guid))
    {
        return intf;
    }
    return getObject(guid);
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! StringBlob !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

void StringBlob::_setUniqueRep(StringRepresentation* uniqueRep)
{
    SLANG_ASSERT(uniqueRep == nullptr || uniqueRep->isUniquelyReferenced());

    m_uniqueRep = uniqueRep;

    m_slice = uniqueRep ? UnownedTerminatedStringSlice(uniqueRep->getData(), uniqueRep->getLength())
                        : UnownedTerminatedStringSlice();
}

/* static */ StringRepresentation* StringBlob::_createUniqueCopy(StringRepresentation* rep)
{
    if (rep)
    {
        // If the length is 0, we can just init as empty
        auto length = rep->getLength();
        if (length == 0)
        {
            return nullptr;
        }
        else
        {
            const UnownedStringSlice slice(rep->getData(), length);
            return StringRepresentation::createWithReference(slice);
        }
    }
    return nullptr;
}

void StringBlob::_setWithCopy(StringRepresentation* rep)
{
    _setUniqueRep(_createUniqueCopy(rep));
}

void StringBlob::_setWithMove(StringRepresentation* rep)
{
    if (rep && !rep->isUniquelyReferenced())
    {
        _setUniqueRep(_createUniqueCopy(rep));
        // We need to release a ref as rep is passed in with the 'current' ref count
        rep->releaseReference();
    }
    else
    {
        _setUniqueRep(rep);
    }
}

/* static */ ComPtr<ISlangBlob> StringBlob::create(const UnownedStringSlice& slice)
{
    StringRepresentation* rep = nullptr;
    if (slice.getLength())
    {
        rep = StringRepresentation::createWithReference(slice);
    }

    auto blob = new StringBlob;

    // rep must be unique at this point
    blob->_setUniqueRep(rep);
    return ComPtr<ISlangBlob>(blob);
}

/* static */ ComPtr<ISlangBlob> StringBlob::create(const String& in)
{
    auto blob = new StringBlob;
    blob->_setWithCopy(in.getStringRepresentation());
    return ComPtr<ISlangBlob>(blob);
}

/* static */ ComPtr<ISlangBlob> StringBlob::moveCreate(String& in)
{
    auto blob = new StringBlob;
    blob->_setWithMove(in.detachStringRepresentation());
    return ComPtr<ISlangBlob>(blob);
}

/* static */ ComPtr<ISlangBlob> StringBlob::moveCreate(String&& in)
{
    auto blob = new StringBlob;
    blob->_setWithMove(in.detachStringRepresentation());
    return ComPtr<ISlangBlob>(blob);
}

StringBlob::~StringBlob()
{
    if (m_uniqueRep)
    {
        delete m_uniqueRep;
    }
}
void* StringBlob::castAs(const SlangUUID& guid)
{
    if (auto intf = getInterface(guid))
    {
        return intf;
    }
    return getObject(guid);
}

void* StringBlob::getObject(const Guid& guid)
{
    // Can allow accessing the contained String
    if (guid == getTypeGuid())
    {
        return this;
    }
    // Can always be accessed as terminated char*
    if (guid == SlangTerminatedChars::getTypeGuid())
    {
        return const_cast<char*>(m_slice.begin());
    }
    return nullptr;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! RawBlob !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

void* RawBlob::castAs(const SlangUUID& guid)
{
    if (auto intf = getInterface(guid))
    {
        return intf;
    }
    return getObject(guid);
}

void* RawBlob::getObject(const Guid& guid)
{
    // If the data has 0 termination, we can return the pointer
    if (guid == SlangTerminatedChars::getTypeGuid() && m_data.isTerminated())
    {
        return (char*)m_data.getData();
    }
    return nullptr;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ScopeBlob !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

void* ScopeBlob::castAs(const SlangUUID& guid)
{
    if (auto intf = getInterface(guid))
    {
        return intf;
    }
    if (auto obj = getObject(guid))
    {
        return obj;
    }

    // If the contained thing is castable, ask it
    if (m_castable)
    {
        return m_castable->castAs(guid);
    }

    return nullptr;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ListBlob !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

void* ListBlob::castAs(const SlangUUID& guid)
{
    if (auto intf = getInterface(guid))
    {
        return intf;
    }
    return getObject(guid);
}

void* ListBlob::getObject(const Guid& guid)
{
    // If the data is terminated return the pointer
    if (guid == SlangTerminatedChars::getTypeGuid())
    {
        const auto count = m_data.getCount();
        if (m_data.getCapacity() > count)
        {
            auto buf = m_data.getBuffer();
            if (buf[count] == 0)
            {
                return (char*)buf;
            }
        }
    }
    return nullptr;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! StaticBlob !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

SlangResult StaticBlob::queryInterface(SlangUUID const& guid, void** outObject)
{
    if (auto intf = getInterface(guid))
    {
        *outObject = intf;
        return SLANG_OK;
    }
    return SLANG_E_NO_INTERFACE;
}

void* StaticBlob::castAs(const SlangUUID& guid)
{
    if (auto intf = getInterface(guid))
    {
        return intf;
    }
    return getObject(guid);
}

ISlangUnknown* StaticBlob::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ISlangBlob::getTypeGuid())
    {
        return static_cast<ISlangBlob*>(this);
    }
    if (guid == ICastable::getTypeGuid())
    {
        return static_cast<ICastable*>(this);
    }
    return nullptr;
}

void* StaticBlob::getObject(const Guid& guid)
{
    SLANG_UNUSED(guid);
    return nullptr;
}

} // namespace Slang
