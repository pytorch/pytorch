// slang-artifact-representation-impl.h
#ifndef SLANG_ARTIFACT_REPRESENTATION_IMPL_H
#define SLANG_ARTIFACT_REPRESENTATION_IMPL_H

#include "../core/slang-com-object.h"
#include "../core/slang-memory-arena.h"
#include "slang-artifact-representation.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang-source-loc.h"

namespace Slang
{

/* A representation of an artifact that is held in a file */
class OSFileArtifactRepresentation : public ComBaseObject, public IOSFileArtifactRepresentation
{
public:
    typedef OSFileArtifactRepresentation ThisType;

    SLANG_COM_BASE_IUNKNOWN_ALL

    // ICastable
    SLANG_NO_THROW void* SLANG_MCALL castAs(const Guid& guid) SLANG_OVERRIDE;

    // IArtifactRepresentation
    SLANG_NO_THROW SlangResult SLANG_MCALL
    createRepresentation(const Guid& typeGuid, ICastable** outCastable) SLANG_OVERRIDE;
    SLANG_NO_THROW bool SLANG_MCALL exists() SLANG_OVERRIDE;

    // IPathArtifactRepresentation
    virtual SLANG_NO_THROW const char* SLANG_MCALL getPath() SLANG_OVERRIDE
    {
        return m_path.getBuffer();
    }
    virtual SLANG_NO_THROW SlangPathType SLANG_MCALL getPathType() SLANG_OVERRIDE
    {
        return SLANG_PATH_TYPE_FILE;
    }
    virtual SLANG_NO_THROW const char* SLANG_MCALL getUniqueIdentity() SLANG_OVERRIDE;

    // IOSFileArtifactRepresentation
    virtual SLANG_NO_THROW Kind SLANG_MCALL getKind() SLANG_OVERRIDE { return m_kind; }
    virtual SLANG_NO_THROW void SLANG_MCALL disown() SLANG_OVERRIDE;
    virtual SLANG_NO_THROW IOSFileArtifactRepresentation* SLANG_MCALL getLockFile() SLANG_OVERRIDE
    {
        return m_lockFile;
    }

    OSFileArtifactRepresentation(
        Kind kind,
        const UnownedStringSlice& path,
        IOSFileArtifactRepresentation* lockFile)
        : m_kind(kind), m_lockFile(lockFile), m_path(path)
    {
    }

    ~OSFileArtifactRepresentation();

    static ComPtr<IOSFileArtifactRepresentation> create(
        Kind kind,
        const UnownedStringSlice& path,
        IOSFileArtifactRepresentation* lockFile)
    {
        return ComPtr<IOSFileArtifactRepresentation>(new ThisType(kind, path, lockFile));
    }

protected:
    void* getInterface(const Guid& uuid);
    void* getObject(const Guid& uuid);

    /// True if the file is owned
    bool _isOwned() const { return Index(m_kind) >= Index(Kind::Owned); }

    static ISlangMutableFileSystem* _getFileSystem();

    Kind m_kind;
    String m_path;
    String m_uniqueIdentity;

    ComPtr<IOSFileArtifactRepresentation> m_lockFile;
    ComPtr<ISlangMutableFileSystem> m_fileSystem;
};

class ExtFileArtifactRepresentation : public ComBaseObject, public IExtFileArtifactRepresentation
{
public:
    typedef ExtFileArtifactRepresentation ThisType;

    SLANG_COM_BASE_IUNKNOWN_ALL

    // ICastable
    SLANG_NO_THROW void* SLANG_MCALL castAs(const Guid& guid) SLANG_OVERRIDE;

    // IArtifactRepresentation
    SLANG_NO_THROW SlangResult SLANG_MCALL
    createRepresentation(const Guid& typeGuid, ICastable** outCastable) SLANG_OVERRIDE;
    SLANG_NO_THROW bool SLANG_MCALL exists() SLANG_OVERRIDE;

    // IPathArtifactRepresentation
    virtual SLANG_NO_THROW const char* SLANG_MCALL getPath() SLANG_OVERRIDE
    {
        return m_path.getBuffer();
    }
    virtual SLANG_NO_THROW SlangPathType SLANG_MCALL getPathType() SLANG_OVERRIDE
    {
        return SLANG_PATH_TYPE_FILE;
    }
    virtual SLANG_NO_THROW const char* SLANG_MCALL getUniqueIdentity() SLANG_OVERRIDE;

    // IExtFileArtifactRepresentation
    virtual SLANG_NO_THROW ISlangFileSystemExt* SLANG_MCALL getFileSystem() SLANG_OVERRIDE
    {
        return m_fileSystem;
    }

    ExtFileArtifactRepresentation(const UnownedStringSlice& path, ISlangFileSystemExt* fileSystem)
        : m_path(path), m_fileSystem(fileSystem)
    {
    }

    static ComPtr<IExtFileArtifactRepresentation> create(
        const UnownedStringSlice& path,
        ISlangFileSystemExt* fileSystem)
    {
        return ComPtr<IExtFileArtifactRepresentation>(new ThisType(path, fileSystem));
    }

protected:
    void* getInterface(const Guid& uuid);
    void* getObject(const Guid& uuid);

    String m_uniqueIdentity;
    String m_path;
    ComPtr<ISlangFileSystemExt> m_fileSystem;
};

class SourceBlobWithPathInfoArtifactRepresentation : public ComBaseObject,
                                                     public IPathArtifactRepresentation
{
public:
    typedef SourceBlobWithPathInfoArtifactRepresentation ThisType;

    SLANG_COM_BASE_IUNKNOWN_ALL

    // ICastable
    SLANG_NO_THROW void* SLANG_MCALL castAs(const Guid& guid) SLANG_OVERRIDE;

    // IArtifactRepresentation
    SLANG_NO_THROW SlangResult SLANG_MCALL
    createRepresentation(const Guid& typeGuid, ICastable** outCastable) SLANG_OVERRIDE;
    SLANG_NO_THROW bool SLANG_MCALL exists() SLANG_OVERRIDE { return false; }

    // IPathArtifactRepresentation
    virtual SLANG_NO_THROW const char* SLANG_MCALL getPath() SLANG_OVERRIDE
    {
        return m_pathInfo.getName().getBuffer();
    }
    virtual SLANG_NO_THROW SlangPathType SLANG_MCALL getPathType() SLANG_OVERRIDE
    {
        return SLANG_PATH_TYPE_FILE;
    }
    virtual SLANG_NO_THROW const char* SLANG_MCALL getUniqueIdentity() SLANG_OVERRIDE
    {
        return m_pathInfo.hasUniqueIdentity() ? m_pathInfo.uniqueIdentity.getBuffer() : nullptr;
    }

    SourceBlobWithPathInfoArtifactRepresentation(const PathInfo& pathInfo, ISlangBlob* sourceBlob)
        : m_pathInfo(pathInfo), m_blob(sourceBlob)
    {
    }

    static ComPtr<IPathArtifactRepresentation> create(
        const PathInfo& pathInfo,
        ISlangBlob* sourceBlob)
    {
        return ComPtr<IPathArtifactRepresentation>(new ThisType(pathInfo, sourceBlob));
    }

protected:
    void* getInterface(const Guid& uuid);
    void* getObject(const Guid& uuid);

    PathInfo m_pathInfo;
    ComPtr<ISlangBlob> m_blob;
};

/* This allows wrapping any object to be an artifact representation.

NOTE! Only allows casting from a single guid. Passing a RefObject across an ABI boundary remains
risky!
*/
class ObjectArtifactRepresentation : public ComBaseObject, public IArtifactRepresentation
{
public:
    SLANG_CLASS_GUID(0xb9d5af57, 0x725b, 0x45f8, {0xac, 0xed, 0x18, 0xf4, 0xa8, 0x4b, 0xf4, 0x73})

    SLANG_COM_BASE_IUNKNOWN_ALL

    // ICastable
    SLANG_NO_THROW void* SLANG_MCALL castAs(const Guid& guid) SLANG_OVERRIDE;
    // IArtifactRepresentation
    SLANG_NO_THROW SlangResult SLANG_MCALL
    createRepresentation(const Guid& guid, ICastable** outCastable) SLANG_OVERRIDE
    {
        SLANG_UNUSED(guid);
        SLANG_UNUSED(outCastable);
        return SLANG_E_NOT_AVAILABLE;
    }
    SLANG_NO_THROW bool SLANG_MCALL exists() SLANG_OVERRIDE { return m_object; }

    ObjectArtifactRepresentation(const Guid& typeGuid, RefObject* obj)
        : m_typeGuid(typeGuid), m_object(obj)
    {
    }

    void* getInterface(const Guid& uuid);
    void* getObject(const Guid& uuid);

    Guid m_typeGuid;            ///< Will return m_object if a cast to m_typeGuid is given
    RefPtr<RefObject> m_object; ///< The object
};

} // namespace Slang

#endif
