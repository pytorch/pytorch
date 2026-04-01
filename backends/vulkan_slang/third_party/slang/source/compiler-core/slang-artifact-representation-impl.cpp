// slang-artifact-representation-impl.cpp
#include "slang-artifact-representation-impl.h"

#include "../core/slang-array-view.h"
#include "../core/slang-castable.h"
#include "../core/slang-file-system.h"
#include "../core/slang-io.h"
#include "../core/slang-type-text-util.h"
#include "slang-artifact-util.h"

namespace Slang
{

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ExtFileArtifactRepresentation !!!!!!!!!!!!!!!!!!!!!!!!!!! */

void* ExtFileArtifactRepresentation::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ICastable::getTypeGuid() ||
        guid == IArtifactRepresentation::getTypeGuid() ||
        guid == IPathArtifactRepresentation::getTypeGuid() ||
        guid == IExtFileArtifactRepresentation::getTypeGuid())
    {
        return static_cast<IExtFileArtifactRepresentation*>(this);
    }
    return nullptr;
}

void* ExtFileArtifactRepresentation::getObject(const Guid& guid)
{
    SLANG_UNUSED(guid);
    return nullptr;
}

void* ExtFileArtifactRepresentation::castAs(const Guid& guid)
{
    if (auto intf = getInterface(guid))
    {
        return intf;
    }
    return getObject(guid);
}

SlangResult ExtFileArtifactRepresentation::createRepresentation(
    const Guid& typeGuid,
    ICastable** outCastable)
{
    // We can convert into a blob only, and only if we have a path
    // If it's referenced by a name only, it's a file that *can't* be loaded as a blob in general.
    if (typeGuid != ISlangBlob::getTypeGuid())
    {
        return SLANG_E_NOT_AVAILABLE;
    }

    ComPtr<ISlangBlob> blob;
    SLANG_RETURN_ON_FAIL(m_fileSystem->loadFile(m_path.getBuffer(), blob.writeRef()));

    *outCastable = CastableUtil::getCastable(blob).detach();
    return SLANG_OK;
}

bool ExtFileArtifactRepresentation::exists()
{
    SlangPathType pathType;
    const auto res = m_fileSystem->getPathType(m_path.getBuffer(), &pathType);
    // It exists if it is a file
    return SLANG_SUCCEEDED(res) && pathType == getPathType();
}

const char* ExtFileArtifactRepresentation::getUniqueIdentity()
{
    if (m_uniqueIdentity.getLength() == 0)
    {
        ComPtr<ISlangBlob> uniqueIdentityBlob;
        if (SLANG_FAILED(m_fileSystem->getFileUniqueIdentity(
                m_path.getBuffer(),
                uniqueIdentityBlob.writeRef())))
        {
            return nullptr;
        }
        m_uniqueIdentity = StringUtil::getString(uniqueIdentityBlob);
    }

    return m_uniqueIdentity.getLength() ? m_uniqueIdentity.getBuffer() : nullptr;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SourceBlobWithPathArtifactRepresentation
 * !!!!!!!!!!!!!!!!!!!!!!!!!!! */

void* SourceBlobWithPathInfoArtifactRepresentation::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ICastable::getTypeGuid() ||
        guid == IArtifactRepresentation::getTypeGuid() ||
        guid == IPathArtifactRepresentation::getTypeGuid())
    {
        return static_cast<IPathArtifactRepresentation*>(this);
    }
    return nullptr;
}

void* SourceBlobWithPathInfoArtifactRepresentation::getObject(const Guid& guid)
{
    SLANG_UNUSED(guid);
    return nullptr;
}

void* SourceBlobWithPathInfoArtifactRepresentation::castAs(const Guid& guid)
{
    if (auto intf = getInterface(guid))
    {
        return intf;
    }
    return getObject(guid);
}

SlangResult SourceBlobWithPathInfoArtifactRepresentation::createRepresentation(
    const Guid& typeGuid,
    ICastable** outCastable)
{
    // We can convert into a blob only.
    if (typeGuid != ISlangBlob::getTypeGuid())
    {
        return SLANG_E_NOT_AVAILABLE;
    }

    if (!m_blob)
    {
        return SLANG_E_NOT_AVAILABLE;
    }

    *outCastable = CastableUtil::getCastable(m_blob).detach();
    return SLANG_OK;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! FileArtifactRepresentation !!!!!!!!!!!!!!!!!!!!!!!!!!! */

void* OSFileArtifactRepresentation::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ICastable::getTypeGuid() ||
        guid == IArtifactRepresentation::getTypeGuid() ||
        guid == IPathArtifactRepresentation::getTypeGuid() ||
        guid == IOSFileArtifactRepresentation::getTypeGuid())
    {
        return static_cast<IOSFileArtifactRepresentation*>(this);
    }
    return nullptr;
}

void* OSFileArtifactRepresentation::getObject(const Guid& guid)
{
    SLANG_UNUSED(guid);
    return nullptr;
}

/* static */ ISlangMutableFileSystem* OSFileArtifactRepresentation::_getFileSystem()
{
    return OSFileSystem::getMutableSingleton();
}

void* OSFileArtifactRepresentation::castAs(const Guid& guid)
{
    if (auto intf = getInterface(guid))
    {
        return intf;
    }
    return getObject(guid);
}

SlangResult OSFileArtifactRepresentation::createRepresentation(
    const Guid& typeGuid,
    ICastable** outCastable)
{
    // We can convert into a blob only, and only if we have a path
    // If it's referenced by a name only, it's a file that *can't* be loaded as a blob in general.
    if (typeGuid != ISlangBlob::getTypeGuid() || m_kind == Kind::NameOnly)
    {
        return SLANG_E_NOT_AVAILABLE;
    }

    ComPtr<ISlangBlob> blob;

    auto fileSystem = _getFileSystem();
    SLANG_RETURN_ON_FAIL(fileSystem->loadFile(m_path.getBuffer(), blob.writeRef()));

    *outCastable = CastableUtil::getCastable(blob).detach();
    return SLANG_OK;
}

bool OSFileArtifactRepresentation::exists()
{
    // TODO(JS):
    // If it's a name only it's hard to know what exists should do. It can't *check* because it
    // relies on the 'system' doing the actual location. We could ask the IArtifactUtil, and that
    // could change the behavior. For now we just assume it does.
    if (m_kind == Kind::NameOnly)
    {
        return true;
    }

    auto fileSystem = _getFileSystem();

    SlangPathType pathType;
    const auto res = fileSystem->getPathType(m_path.getBuffer(), &pathType);

    // It exists if it is a file
    return SLANG_SUCCEEDED(res) && pathType == SLANG_PATH_TYPE_FILE;
}

const char* OSFileArtifactRepresentation::getUniqueIdentity()
{
    if (m_uniqueIdentity.getLength() == 0)
    {
        auto fileSystem = _getFileSystem();

        ComPtr<ISlangBlob> uniqueIdentityBlob;
        if (SLANG_FAILED(fileSystem->getFileUniqueIdentity(
                m_path.getBuffer(),
                uniqueIdentityBlob.writeRef())))
        {
            return nullptr;
        }
        m_uniqueIdentity = StringUtil::getString(uniqueIdentityBlob);
    }

    return m_uniqueIdentity.getLength() ? m_uniqueIdentity.getBuffer() : nullptr;
}

void OSFileArtifactRepresentation::disown()
{
    if (_isOwned())
    {
        m_kind = Kind::Reference;
    }
}

OSFileArtifactRepresentation::~OSFileArtifactRepresentation()
{
    if (_isOwned())
    {
        auto fileSystem = _getFileSystem();
        fileSystem->remove(m_path.getBuffer());
    }
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PostEmitMetadataArtifactRepresentation !!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

void* ObjectArtifactRepresentation::castAs(const Guid& guid)
{

    if (auto ptr = getInterface(guid))
    {
        return ptr;
    }
    return getObject(guid);
}

void* ObjectArtifactRepresentation::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ICastable::getTypeGuid() ||
        guid == IArtifactRepresentation::getTypeGuid())
    {
        return static_cast<IArtifactRepresentation*>(this);
    }
    return nullptr;
}

void* ObjectArtifactRepresentation::getObject(const Guid& guid)
{
    if (guid == getTypeGuid())
    {
        return this;
    }

    // If matches the guid saved in the object, we return that
    if (m_object && m_typeGuid == guid)
    {
        return m_object;
    }

    return nullptr;
}

} // namespace Slang
