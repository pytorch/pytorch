// slang-castable.cpp
#include "slang-castable.h"

namespace Slang
{

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CastableUtil !!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* static */ ComPtr<ICastable> CastableUtil::getCastable(ISlangUnknown* unk)
{
    SLANG_ASSERT(unk);
    ComPtr<ICastable> castable;
    if (SLANG_SUCCEEDED(unk->queryInterface(SLANG_IID_PPV_ARGS(castable.writeRef()))))
    {
        SLANG_ASSERT(castable);
    }
    else
    {
        castable = new UnknownCastableAdapter(unk);
    }
    return castable;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! UnknownCastableAdapter !!!!!!!!!!!!!!!!!!!!!!!!!!! */

void* UnknownCastableAdapter::castAs(const Guid& guid)
{
    if (auto intf = getInterface(guid))
    {
        return intf;
    }
    if (auto obj = getObject(guid))
    {
        return obj;
    }

    if (m_found && guid == m_foundGuid)
    {
        return m_found;
    }

    ComPtr<ISlangUnknown> cast;
    if (SLANG_SUCCEEDED(m_contained->queryInterface(guid, (void**)cast.writeRef())) && cast)
    {
        // Save the interface in the cache
        m_found = cast;
        m_foundGuid = guid;

        return cast;
    }
    return nullptr;
}

void* UnknownCastableAdapter::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ICastable::getTypeGuid() ||
        guid == IUnknownCastableAdapter::getTypeGuid())
    {
        return static_cast<IUnknownCastableAdapter*>(this);
    }
    return nullptr;
}

void* UnknownCastableAdapter::getObject(const Guid& guid)
{
    SLANG_UNUSED(guid);
    return nullptr;
}

} // namespace Slang
