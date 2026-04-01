#include "slang-file-system.h"

#include "../core/slang-io.h"
#include "../core/slang-string-util.h"
#include "slang-com-ptr.h"

namespace Slang
{

SLANG_FORCE_INLINE static SlangResult _checkExt(FileSystemStyle style)
{
    return Index(style) >= Index(FileSystemStyle::Ext) ? SLANG_OK : SLANG_E_NOT_IMPLEMENTED;
}
SLANG_FORCE_INLINE static SlangResult _checkMutable(FileSystemStyle style)
{
    return Index(style) >= Index(FileSystemStyle::Mutable) ? SLANG_OK : SLANG_E_NOT_IMPLEMENTED;
}

SLANG_FORCE_INLINE static bool _canCast(FileSystemStyle style, const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ISlangCastable::getTypeGuid() ||
        guid == ISlangFileSystem::getTypeGuid())
    {
        return true;
    }
    else if (guid == ISlangFileSystemExt::getTypeGuid())
    {
        return Index(style) >= Index(FileSystemStyle::Ext);
    }
    else if (guid == ISlangMutableFileSystem::getTypeGuid())
    {
        return Index(style) >= Index(FileSystemStyle::Mutable);
    }
    return false;
}

static FileSystemStyle _getFileSystemStyle(ISlangFileSystem* system, ComPtr<ISlangFileSystem>& out)
{
    SLANG_ASSERT(system);

    FileSystemStyle style = FileSystemStyle::Load;

    if (SLANG_SUCCEEDED(
            system->queryInterface(ISlangMutableFileSystem::getTypeGuid(), (void**)out.writeRef())))
    {
        style = FileSystemStyle::Mutable;
    }
    else if (SLANG_SUCCEEDED(system->queryInterface(
                 ISlangFileSystemExt::getTypeGuid(),
                 (void**)out.writeRef())))
    {
        style = FileSystemStyle::Ext;
    }
    else
    {
        style = FileSystemStyle::Load;
        out = system;
    }

    SLANG_ASSERT(out);
    return style;
}

// Calcuate a combined path, just using Path:: string processing
static SlangResult _calcCombinedPath(
    SlangPathType fromPathType,
    const char* fromPath,
    const char* path,
    ISlangBlob** pathOut)
{
    String relPath;
    switch (fromPathType)
    {
    case SLANG_PATH_TYPE_FILE:
        {
            relPath = Path::combine(Path::getParentDirectory(fromPath), path);
            break;
        }
    case SLANG_PATH_TYPE_DIRECTORY:
        {
            relPath = Path::combine(fromPath, path);
            break;
        }
    }

    *pathOut = StringUtil::createStringBlob(relPath).detach();
    return SLANG_OK;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!! OSFileSystem !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

/* static */ OSFileSystem OSFileSystem::g_load(FileSystemStyle::Load);
/* static */ OSFileSystem OSFileSystem::g_ext(FileSystemStyle::Ext);
/* static */ OSFileSystem OSFileSystem::g_mutable(FileSystemStyle::Mutable);

void* OSFileSystem::castAs(const Guid& guid)
{
    if (auto ptr = getInterface(guid))
    {
        return ptr;
    }
    return getObject(guid);
}

ISlangUnknown* OSFileSystem::getInterface(const Guid& guid)
{
    return _canCast(m_style, guid) ? static_cast<ISlangFileSystem*>(this) : nullptr;
}

void* OSFileSystem::getObject(const Guid& guid)
{
    SLANG_UNUSED(guid);
    return nullptr;
}

static String _fixPathDelimiters(const char* pathIn)
{
#if SLANG_WINDOWS_FAMILY
    return pathIn;
#else
    // To allow windows style \ delimiters on other platforms, we convert to our standard delimiter
    String path(pathIn);
    return StringUtil::calcCharReplaced(pathIn, '\\', Path::kPathDelimiter);
#endif
}

SlangResult OSFileSystem::getFileUniqueIdentity(const char* pathIn, ISlangBlob** outUniqueIdentity)
{
    SLANG_RETURN_ON_FAIL(_checkExt(m_style));

    // By default we use the canonical path to uniquely identify a file
    return getPath(PathKind::Canonical, pathIn, outUniqueIdentity);
}

SlangResult OSFileSystem::getPath(PathKind pathKind, const char* path, ISlangBlob** outPath)
{
    SLANG_RETURN_ON_FAIL(_checkExt(m_style));

    switch (pathKind)
    {
    case PathKind::OperatingSystem:
    case PathKind::Display:
        {
            // It's possible canonical path fail...
            if (SLANG_SUCCEEDED(getPath(PathKind::Canonical, path, outPath)))
            {
                return SLANG_OK;
            }
            // If so try simplified
            return getPath(PathKind::Simplified, path, outPath);
        }
    case PathKind::Canonical:
        {
            String canonicalPath;
            SLANG_RETURN_ON_FAIL(Path::getCanonical(_fixPathDelimiters(path), canonicalPath));
            *outPath = StringUtil::createStringBlob(canonicalPath).detach();
            return SLANG_OK;
        }
    case PathKind::Simplified:
        {
            String simplifiedPath = Path::simplify(path);
            *outPath = StringUtil::createStringBlob(simplifiedPath).detach();
            return SLANG_OK;
        }
    }

    return SLANG_E_NOT_AVAILABLE;
}

SlangResult OSFileSystem::calcCombinedPath(
    SlangPathType fromPathType,
    const char* fromPath,
    const char* path,
    ISlangBlob** pathOut)
{
    SLANG_RETURN_ON_FAIL(_checkExt(m_style));

    // Don't need to fix delimiters - because combine path handles both path delimiter types
    return _calcCombinedPath(fromPathType, fromPath, path, pathOut);
}

SlangResult SLANG_MCALL OSFileSystem::getPathType(const char* pathIn, SlangPathType* pathTypeOut)
{
    SLANG_RETURN_ON_FAIL(_checkExt(m_style));

    return Path::getPathType(_fixPathDelimiters(pathIn), pathTypeOut);
}


SlangResult OSFileSystem::loadFile(char const* pathIn, ISlangBlob** outBlob)
{
    // Default implementation that uses the `core` libraries facilities for talking to the OS
    // filesystem.
    //
    // TODO: we might want to conditionally compile these in, so that
    // a user could create a build of Slang that doesn't include any OS
    // filesystem calls.

    const String path = _fixPathDelimiters(pathIn);
    if (!File::exists(path))
    {
        return SLANG_E_NOT_FOUND;
    }

    ScopedAllocation alloc;
    SLANG_RETURN_ON_FAIL(File::readAllBytes(path, alloc));
    *outBlob = RawBlob::moveCreate(alloc).detach();
    return SLANG_OK;
}

SlangResult OSFileSystem::enumeratePathContents(
    const char* path,
    FileSystemContentsCallBack callback,
    void* userData)
{
    SLANG_RETURN_ON_FAIL(_checkExt(m_style));

    struct Visitor : Path::Visitor
    {
        void accept(Path::Type type, const UnownedStringSlice& filename) SLANG_OVERRIDE
        {
            m_buffer.clear();
            m_buffer.append(filename);

            SlangPathType pathType;
            switch (type)
            {
            case Path::Type::File:
                pathType = SLANG_PATH_TYPE_FILE;
                break;
            case Path::Type::Directory:
                pathType = SLANG_PATH_TYPE_DIRECTORY;
                break;
            default:
                return;
            }

            m_callback(pathType, m_buffer.getBuffer(), m_userData);
        }

        Visitor(FileSystemContentsCallBack callback, void* userData)
            : m_callback(callback), m_userData(userData)
        {
        }
        StringBuilder m_buffer;
        FileSystemContentsCallBack m_callback;
        void* m_userData;
    };

    Visitor visitor(callback, userData);
    Path::find(path, nullptr, &visitor);

    return SLANG_OK;
}

SlangResult OSFileSystem::saveFile(const char* pathIn, const void* data, size_t size)
{
    SLANG_RETURN_ON_FAIL(_checkMutable(m_style));
    const String path = _fixPathDelimiters(pathIn);
    FileStream stream;
    SLANG_RETURN_ON_FAIL(
        stream.init(pathIn, FileMode::Create, FileAccess::Write, FileShare::ReadWrite));
    SLANG_RETURN_ON_FAIL(stream.write(data, size));
    return SLANG_OK;
}

SlangResult OSFileSystem::saveFileBlob(const char* path, ISlangBlob* dataBlob)
{
    if (!dataBlob)
    {
        return SLANG_E_INVALID_ARG;
    }
    return saveFile(path, dataBlob->getBufferPointer(), dataBlob->getBufferSize());
}

SlangResult OSFileSystem::remove(const char* path)
{
    SLANG_RETURN_ON_FAIL(_checkMutable(m_style));
    return Path::remove(path);
}

SlangResult OSFileSystem::createDirectory(const char* path)
{
    SLANG_RETURN_ON_FAIL(_checkMutable(m_style));
    return Path::createDirectory(path);
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CacheFileSystem !!!!!!!!!!!!!!!!!!!!!!!!!!!

/* static */ const Result CacheFileSystem::s_compressedResultToResult[] = {
    SLANG_E_UNINITIALIZED,
    SLANG_OK,            ///< Ok
    SLANG_E_NOT_FOUND,   ///< File not found
    SLANG_E_CANNOT_OPEN, ///< CannotOpen,
    SLANG_FAIL,          ///< Fail
};

/* static */ CacheFileSystem::CompressedResult CacheFileSystem::toCompressedResult(Result res)
{
    if (SLANG_SUCCEEDED(res))
    {
        return CompressedResult::Ok;
    }
    switch (res)
    {
    case SLANG_E_CANNOT_OPEN:
        return CompressedResult::CannotOpen;
    case SLANG_E_NOT_FOUND:
        return CompressedResult::NotFound;
    default:
        return CompressedResult::Fail;
    }
}

void* CacheFileSystem::castAs(const Guid& guid)
{
    if (auto ptr = getInterface(guid))
    {
        return ptr;
    }
    return getObject(guid);
}

void* CacheFileSystem::getInterface(const Guid& guid)
{
    if (_canCast(FileSystemStyle::Ext, guid))
    {
        return static_cast<ISlangFileSystemExt*>(this);
    }
    return nullptr;
}

void* CacheFileSystem::getObject(const Guid& guid)
{
    if (guid == CacheFileSystem::getTypeGuid())
    {
        return this;
    }
    return nullptr;
}

CacheFileSystem::CacheFileSystem(
    ISlangFileSystem* fileSystem,
    UniqueIdentityMode uniqueIdentityMode,
    PathStyle pathStyle)
{
    setInnerFileSystem(fileSystem, uniqueIdentityMode, pathStyle);
}

CacheFileSystem::~CacheFileSystem()
{
    for (const auto& [_, pathInfo] : m_uniqueIdentityMap)
        delete pathInfo;
}

void CacheFileSystem::setInnerFileSystem(
    ISlangFileSystem* fileSystem,
    UniqueIdentityMode uniqueIdentityMode,
    PathStyle pathStyle)
{
    m_fileSystem = fileSystem;

    m_uniqueIdentityMode = uniqueIdentityMode;
    m_pathStyle = pathStyle;

    m_fileSystemExt.setNull();

    if (fileSystem)
    {
        // Try to get the more sophisticated interface
        fileSystem->queryInterface(SLANG_IID_PPV_ARGS(m_fileSystemExt.writeRef()));
    }

    // Determine how paths map
    m_osPathKind = m_fileSystemExt ? m_fileSystemExt->getOSPathKind() : OSPathKind::None;

    switch (m_uniqueIdentityMode)
    {
    case UniqueIdentityMode::Default:
    case UniqueIdentityMode::FileSystemExt:
        {
            // If it's not a complete file system, we will default to SimplifyAndHash style by
            // default
            m_uniqueIdentityMode = m_fileSystemExt ? UniqueIdentityMode::FileSystemExt
                                                   : UniqueIdentityMode::SimplifyPathAndHash;
            break;
        }
    default:
        break;
    }

    if (pathStyle == PathStyle::Default)
    {
        // We'll assume it's simplify-able
        m_pathStyle = PathStyle::Simplifiable;
        // If we have fileSystemExt, we defer to that
        if (m_fileSystemExt)
        {
            // We just defer to the m_fileSystem
            m_pathStyle = PathStyle::FileSystemExt;
        }
    }

    // It can't be default
    SLANG_ASSERT(m_uniqueIdentityMode != UniqueIdentityMode::Default);
}

void CacheFileSystem::clearCache()
{
    for (const auto& [_, pathInfo] : m_uniqueIdentityMap)
        delete pathInfo;

    m_uniqueIdentityMap.clear();
    m_pathMap.clear();

    if (m_fileSystemExt)
    {
        m_fileSystemExt->clearCache();
    }
}


// Determines if we can simplify a path for a given mode
static bool _canSimplifyPath(CacheFileSystem::UniqueIdentityMode mode)
{
    typedef CacheFileSystem::UniqueIdentityMode UniqueIdentityMode;
    switch (mode)
    {
    case UniqueIdentityMode::SimplifyPath:
    case UniqueIdentityMode::SimplifyPathAndHash:
        {
            return true;
        }
    default:
        {
            return false;
        }
    }
}

SlangResult CacheFileSystem::enumeratePathContents(
    const char* path,
    FileSystemContentsCallBack callback,
    void* userData)
{
    if (m_fileSystemExt)
    {
        return m_fileSystemExt->enumeratePathContents(path, callback, userData);
    }

    // Okay.. the contents of the 'cache' *is* the filesystem. So lets iterate over that
    // This will win no prizes for efficiency, but that is unlikely to matter for typical usage

    if (!_canSimplifyPath(m_uniqueIdentityMode))
    {
        // As it stands if we can't simplify paths, it's kind of hard to make this
        // all work. As we use the simplified path cache
        return SLANG_E_NOT_IMPLEMENTED;
    }

    // Simplify the path
    String simplifiedPath = Path::simplify(path);

    // If the simplified path is just a . then we don't have any prefix
    if (simplifiedPath == ".")
    {
        simplifiedPath = "";
    }

    for (const auto& [currentPath, pathInfo] : m_pathMap)
    {
        // NOTE! The currentPath can be a *non* simplified path (the m_pathMap is the cache of paths
        // simplified and other to a file/directory) Also note that there will always be the
        // simplified version of the path in cache.

        // If it doesn't start with simplified path, then it can't be a hit
        if (!currentPath.startsWith(simplifiedPath))
        {
            continue;
        }

        UnownedStringSlice remaining(
            currentPath.getBuffer() + simplifiedPath.getLength(),
            currentPath.end());

        // If it starts with a / delimiter strip it
        if (remaining.getLength() > 0 && remaining[0] == '/')
        {
            remaining = UnownedStringSlice(remaining.begin() + 1, remaining.end());
        }

        // If it has a path separator then it's either not simplified - so we ignore (we only want
        // to invoke on the simplified path version as there is only one of these for every
        // PathInfo) or it is a child file/directory, and so we ignore that too.
        if (remaining.indexOf('/') >= 0 || remaining.indexOf('\\') >= 0)
        {
            continue;
        }

        // We *know* that remaining comes from the end of currentPath .We also know currentPath is
        // zero terminated. So we can just use (normally this would be a problem because
        // UnownedStringSlice is generally *not* followed by zero termination.
        const char* foundPath = remaining.begin();
        // Let's check that fact...
        SLANG_ASSERT(foundPath[remaining.getLength()] == 0);

        SlangPathType pathType;
        if (SLANG_FAILED(_getPathType(pathInfo, currentPath.getBuffer(), &pathType)))
        {
            continue;
        }

        callback(pathType, foundPath, userData);
    }

    return SLANG_OK;
}


SlangResult CacheFileSystem::_calcUniqueIdentity(
    const String& path,
    String& outUniqueIdentity,
    ComPtr<ISlangBlob>& outFileContents)
{
    switch (m_uniqueIdentityMode)
    {
    case UniqueIdentityMode::FileSystemExt:
        {
            // Try getting the uniqueIdentity by asking underlying file system
            ComPtr<ISlangBlob> uniqueIdentity;
            SLANG_RETURN_ON_FAIL(m_fileSystemExt->getFileUniqueIdentity(
                path.getBuffer(),
                uniqueIdentity.writeRef()));
            // Get the path as a string
            outUniqueIdentity = StringUtil::getString(uniqueIdentity);
            return SLANG_OK;
        }
    case UniqueIdentityMode::Path:
        {
            outUniqueIdentity = path;
            return SLANG_OK;
        }
    case UniqueIdentityMode::SimplifyPath:
        {
            outUniqueIdentity = Path::simplify(path);
            // If it still has relative elements can't uniquely identify, so give up
            return Path::hasRelativeElement(outUniqueIdentity) ? SLANG_FAIL : SLANG_OK;
        }
    case UniqueIdentityMode::SimplifyPathAndHash:
    case UniqueIdentityMode::Hash:
        {
            // If m_uniqueIdentityMode is SimplifyPathAndHash, the path will already be simplified
            // before this function is hit (and it hasn't been found via path lookup). That being
            // the case only option left is to 'hash' (or fallback to backing impls uniqueIdentity
            // impl)

            // If we don't have a file system -> assume cannot be found
            if (m_fileSystem == nullptr)
            {
                return SLANG_E_NOT_FOUND;
            }

            // First attempt to load as a file
            Result res = m_fileSystem->loadFile(path.getBuffer(), outFileContents.writeRef());

            // If it succeeded but there is no contents, then make the result NOT_FOUND
            res = (SLANG_SUCCEEDED(res) && outFileContents == nullptr) ? SLANG_E_NOT_FOUND : res;

            // If that failed, we may be able to do something if m_fileSystemExt is available
            if (SLANG_FAILED(res))
            {
                // If we have m_fileSystemExt interface we can just use it's implementation, as a
                // fallback. Doing so will mean the uniqueIdentity will work if say it's a directory
                if (m_fileSystemExt)
                {
                    ComPtr<ISlangBlob> uniqueIdentity;
                    SLANG_RETURN_ON_FAIL(m_fileSystemExt->getFileUniqueIdentity(
                        path.getBuffer(),
                        uniqueIdentity.writeRef()));
                    // Get the path as a string
                    outUniqueIdentity = StringUtil::getString(uniqueIdentity);
                    return SLANG_OK;
                }

                // If we can't access as a file (or use the backing implementations impl), we are in
                // a tricky situation. The ISlangFileSystem interface provides no way to determine
                // if the path is a directory for example - so there is no way of determining if
                // something along the path exists.
                //
                // So we just return the error.
                return res;
            }

            // Calculate the hash on the contents
            const StableHashCode64 hash = getStableHashCode64(
                (const char*)outFileContents->getBufferPointer(),
                outFileContents->getBufferSize());

            String hashString = Path::getFileName(path);
            hashString = hashString.toLower();

            hashString.append(':');

            // The uniqueIdentity is a combination of name and hash
            hashString.append(hash);

            outUniqueIdentity = hashString;
            return SLANG_OK;
        }
    }

    return SLANG_FAIL;
}

CacheFileSystem::PathInfo* CacheFileSystem::_resolveUniqueIdentityCacheInfo(const String& path)
{
    // Use the path to produce uniqueIdentity information
    ComPtr<ISlangBlob> fileContents;
    String uniqueIdentity;

    SlangResult res = _calcUniqueIdentity(path, uniqueIdentity, fileContents);
    if (SLANG_FAILED(res))
    {
        // Was not able to create a uniqueIdentity - return failure as nullptr
        return nullptr;
    }

    // Now try looking up by uniqueIdentity path. If not found, add a new result
    PathInfo* pathInfo = nullptr;
    if (!m_uniqueIdentityMap.tryGetValue(uniqueIdentity, pathInfo))
    {
        // Create with found uniqueIdentity
        pathInfo = new PathInfo(uniqueIdentity);
        m_uniqueIdentityMap.add(uniqueIdentity, pathInfo);
    }

    // At this point they must have same uniqueIdentity
    SLANG_ASSERT(pathInfo->getUniqueIdentity() == uniqueIdentity);

    // If we have the file contents (because of calc-ing uniqueIdentity), and there isn't a read
    // file blob already store the data as if read, so doesn't get read again
    if (fileContents && !pathInfo->m_fileBlob)
    {
        pathInfo->m_fileBlob = fileContents;
        pathInfo->m_loadFileResult = CompressedResult::Ok;
    }

    return pathInfo;
}

CacheFileSystem::PathInfo* CacheFileSystem::_resolveSimplifiedPathCacheInfo(const String& path)
{
    // If we can simplify the path, try looking up in path cache with simplified path (as long as
    // it's different!)
    if (_canSimplifyPath(m_uniqueIdentityMode))
    {
        const String simplifiedPath = Path::simplify(path);
        // Only lookup if the path is different - because otherwise will recurse forever...
        if (simplifiedPath != path)
        {
            // This is a recursive call - and will ensure the simplified path is added to the cache
            return _resolvePathCacheInfo(simplifiedPath);
        }
    }

    return _resolveUniqueIdentityCacheInfo(path);
}

CacheFileSystem::PathInfo* CacheFileSystem::_resolvePathCacheInfo(const String& path)
{
    // Lookup in path cache
    PathInfo* pathInfo;
    if (m_pathMap.tryGetValue(path, pathInfo))
    {
        // Found so done
        return pathInfo;
    }

    // Try getting or creating taking into account possible path simplification
    pathInfo = _resolveSimplifiedPathCacheInfo(path);
    // Always add the result to the path cache (even if null)
    m_pathMap.add(path, pathInfo);
    return pathInfo;
}

SlangResult CacheFileSystem::loadFile(char const* pathIn, ISlangBlob** blobOut)
{
    *blobOut = nullptr;
    String path(pathIn);
    PathInfo* info = _resolvePathCacheInfo(path);
    if (!info)
    {
        return SLANG_FAIL;
    }

    if (info->m_loadFileResult == CompressedResult::Uninitialized)
    {
        info->m_loadFileResult = toCompressedResult(
            m_fileSystem->loadFile(path.getBuffer(), info->m_fileBlob.writeRef()));
    }

    *blobOut = info->m_fileBlob;
    if (*blobOut)
    {
        (*blobOut)->addRef();
    }
    return toResult(info->m_loadFileResult);
}

SlangResult CacheFileSystem::getFileUniqueIdentity(const char* path, ISlangBlob** outUniqueIdentity)
{
    *outUniqueIdentity = nullptr;
    PathInfo* info = _resolvePathCacheInfo(path);
    if (!info || info->m_uniqueIdentity.getLength() <= 0)
    {
        return SLANG_E_NOT_FOUND;
    }

    *outUniqueIdentity = StringBlob::create(info->m_uniqueIdentity).detach();
    return SLANG_OK;
}

SlangResult CacheFileSystem::calcCombinedPath(
    SlangPathType fromPathType,
    const char* fromPath,
    const char* path,
    ISlangBlob** pathOut)
{
    // Just defer to contained implementation
    switch (m_pathStyle)
    {
    case PathStyle::FileSystemExt:
        {
            return m_fileSystemExt->calcCombinedPath(fromPathType, fromPath, path, pathOut);
        }
    default:
        {
            // Just use the default implementation
            return _calcCombinedPath(fromPathType, fromPath, path, pathOut);
        }
    }
}

SlangResult CacheFileSystem::_getPathType(
    PathInfo* info,
    const char* inPath,
    SlangPathType* outPathType)
{
    if (info->m_getPathTypeResult == CompressedResult::Uninitialized)
    {
        if (m_fileSystemExt)
        {
            info->m_getPathTypeResult =
                toCompressedResult(m_fileSystemExt->getPathType(inPath, &info->m_pathType));
        }
        else
        {
            // Okay try to load the file
            if (info->m_loadFileResult == CompressedResult::Uninitialized)
            {
                info->m_loadFileResult =
                    toCompressedResult(m_fileSystem->loadFile(inPath, info->m_fileBlob.writeRef()));
            }

            // Make the getPathResult the same as the load result
            info->m_getPathTypeResult = info->m_loadFileResult;
            // Just set to file... the result is what matters in this case
            info->m_pathType = SLANG_PATH_TYPE_FILE;
        }
    }

    *outPathType = info->m_pathType;
    return toResult(info->m_getPathTypeResult);
}

SlangResult CacheFileSystem::getPathType(const char* inPath, SlangPathType* outPathType)
{
    PathInfo* info = _resolvePathCacheInfo(inPath);
    if (!info)
    {
        return SLANG_E_NOT_FOUND;
    }

    return _getPathType(info, inPath, outPathType);
}

SlangResult CacheFileSystem::getPath(PathKind kind, const char* path, ISlangBlob** outPath)
{
    switch (kind)
    {
    case PathKind::Simplified:
        return _getSimplifiedPath(path, outPath);
    case PathKind::Canonical:
        return _getCanonicalPath(path, outPath);
    default:
        break;
    }

    if (m_fileSystemExt)
    {
        return m_fileSystemExt->getPath(kind, path, outPath);
    }

    // If we don't have a fileSystem, we can try the canonical path
    if (SLANG_SUCCEEDED(getPath(PathKind::Canonical, path, outPath)))
    {
        return SLANG_OK;
    }
    // Else we can try simplified
    return getPath(PathKind::Simplified, path, outPath);
}

SlangResult CacheFileSystem::_getSimplifiedPath(const char* path, ISlangBlob** outSimplifiedPath)
{
    // If we have a ISlangFileSystemExt we can just pass on the request to it
    switch (m_pathStyle)
    {
    case PathStyle::FileSystemExt:
        {
            return m_fileSystemExt->getPath(PathKind::Simplified, path, outSimplifiedPath);
        }
    case PathStyle::Simplifiable:
        {
            String simplifiedPath = Path::simplify(path);
            *outSimplifiedPath = StringUtil::createStringBlob(simplifiedPath).detach();
            return SLANG_OK;
        }
    default:
        return SLANG_E_NOT_IMPLEMENTED;
    }
}

SlangResult CacheFileSystem::_getCanonicalPath(const char* path, ISlangBlob** outCanonicalPath)
{
    *outCanonicalPath = nullptr;

    // A file must exist to get a canonical path...
    PathInfo* info = _resolvePathCacheInfo(path);
    if (!info)
    {
        return SLANG_E_NOT_FOUND;
    }

    // We don't have this -> so read it ...
    if (info->m_getCanonicalPathResult == CompressedResult::Uninitialized)
    {
        if (!m_fileSystemExt)
        {
            return SLANG_E_NOT_IMPLEMENTED;
        }

        // Try getting the canonicalPath by asking underlying file system
        ComPtr<ISlangBlob> canonicalPathBlob;
        SlangResult res =
            m_fileSystemExt->getPath(PathKind::Canonical, path, canonicalPathBlob.writeRef());

        if (SLANG_SUCCEEDED(res))
        {
            // Get the path as a string
            info->m_canonicalPath = StringUtil::getString(canonicalPathBlob);
            if (info->m_canonicalPath.getLength() <= 0)
            {
                res = SLANG_FAIL;
            }
        }

        // Save the result
        info->m_getCanonicalPathResult = toCompressedResult(res);
    }

    // Create the blob
    if (info->m_canonicalPath.getLength())
    {
        *outCanonicalPath = StringBlob::create(info->m_canonicalPath).detach();
    }

    return SLANG_OK;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  RelativeFileSystem  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

RelativeFileSystem::RelativeFileSystem(
    ISlangFileSystem* fileSystem,
    const String& relativePath,
    bool stripPath)
    : m_relativePath(relativePath), m_stripPath(stripPath)
{
    m_style = _getFileSystemStyle(fileSystem, m_fileSystem);

    m_osPathKind = OSPathKind::None;

    ComPtr<ISlangFileSystemExt> ext;
    if (SLANG_SUCCEEDED(fileSystem->queryInterface(SLANG_IID_PPV_ARGS(ext.writeRef()))))
    {
        m_osPathKind = ext->getOSPathKind();

        // If it's direct, but we have a relative path, "operating system" should work
        if (m_osPathKind == OSPathKind::Direct && relativePath.getLength())
        {
            m_osPathKind = OSPathKind::OperatingSystem;
        }
    }
}

ISlangUnknown* RelativeFileSystem::getInterface(const Guid& guid)
{
    return _canCast(m_style, guid) ? static_cast<ISlangMutableFileSystem*>(this) : nullptr;
}

void* RelativeFileSystem::getObject(const Guid& guid)
{
    SLANG_UNUSED(guid);
    return nullptr;
}

void* RelativeFileSystem::castAs(const Guid& guid)
{
    if (auto ptr = getInterface(guid))
    {
        return ptr;
    }
    return getObject(guid);
}

SlangResult RelativeFileSystem::_calcCombinedPathInner(
    SlangPathType fromPathType,
    const char* fromPath,
    const char* path,
    ISlangBlob** outPath)
{
    ISlangFileSystemExt* fileSystem = _getExt();
    if (fileSystem)
    {
        return fileSystem->calcCombinedPath(fromPathType, fromPath, path, outPath);
    }
    else
    {
        return _calcCombinedPath(fromPathType, fromPath, path, outPath);
    }
}

SlangResult RelativeFileSystem::_getCanonicalPath(const char* path, String& outPath)
{
    if (m_stripPath)
    {
        // We are just using the filename. There is no path that could go outside of the the
        // relative path so we can use as is
        outPath = Path::getFileName(path);
    }
    else
    {
        // NOTE that we don't want the canonical path to be absolute with a leading "/"
        // because paths specified which aren't absolute, would produce a different path.
        //
        // Ie we want (and get with these options)
        // "a" -> "a"
        // "/a" -> "a".
        //
        // If we allowed the root to be included then...
        // "a" -> "a"
        // "/a" -> "/a"
        //
        // Two identical paths would match to different paths, which wouldn't be canonical.
        //
        // This could be fixed by making all paths absolute with '/' too, but it's easier to just
        // make all not have "/"

        StringBuilder canonicalPath;
        // We want the input path to be local to this file system
        SLANG_RETURN_ON_FAIL(
            Path::simplify(path, Path::SimplifyStyle::AbsoluteOnlyAndNoRoot, canonicalPath));
        outPath = canonicalPath;
    }
    return SLANG_OK;
}

SlangResult RelativeFileSystem::_getFixedPath(const char* path, String& outPath)
{
    ComPtr<ISlangBlob> blob;

    String canonicalPath;
    SLANG_RETURN_ON_FAIL(_getCanonicalPath(path, canonicalPath));

    SLANG_RETURN_ON_FAIL(_calcCombinedPathInner(
        SLANG_PATH_TYPE_DIRECTORY,
        m_relativePath.getBuffer(),
        canonicalPath.getBuffer(),
        blob.writeRef()));
    outPath = StringUtil::getString(blob);

    return SLANG_OK;
}

SlangResult RelativeFileSystem::loadFile(char const* path, ISlangBlob** outBlob)
{
    String fixedPath;
    SLANG_RETURN_ON_FAIL(_getFixedPath(path, fixedPath));
    return m_fileSystem->loadFile(fixedPath.getBuffer(), outBlob);
}

SlangResult RelativeFileSystem::getFileUniqueIdentity(
    const char* path,
    ISlangBlob** outUniqueIdentity)
{
    auto fileSystem = _getExt();
    if (!fileSystem)
        return SLANG_E_NOT_IMPLEMENTED;

    String fixedPath;
    SLANG_RETURN_ON_FAIL(_getFixedPath(path, fixedPath));
    return fileSystem->getFileUniqueIdentity(fixedPath.getBuffer(), outUniqueIdentity);
}

SlangResult RelativeFileSystem::calcCombinedPath(
    SlangPathType fromPathType,
    const char* fromPath,
    const char* path,
    ISlangBlob** outPath)
{
    auto fileSystem = _getExt();
    if (!fileSystem)
        return SLANG_E_NOT_IMPLEMENTED;

    String fixedFromPath;
    SLANG_RETURN_ON_FAIL(_getFixedPath(fromPath, fixedFromPath));

    return fileSystem->calcCombinedPath(fromPathType, fixedFromPath.getBuffer(), path, outPath);
}

SlangResult RelativeFileSystem::getPathType(const char* path, SlangPathType* outPathType)
{
    auto fileSystem = _getExt();
    if (!fileSystem)
        return SLANG_E_NOT_IMPLEMENTED;

    String fixedPath;
    SLANG_RETURN_ON_FAIL(_getFixedPath(path, fixedPath));
    return fileSystem->getPathType(fixedPath.getBuffer(), outPathType);
}

SlangResult RelativeFileSystem::getPath(PathKind kind, const char* path, ISlangBlob** outPath)
{
    auto fileSystem = _getExt();
    if (!fileSystem)
        return SLANG_E_NOT_IMPLEMENTED;

    switch (kind)
    {
    case PathKind::Simplified:
        {
            return fileSystem->getPath(kind, path, outPath);
        }
    case PathKind::Display:
        {
            // If not backed by OS, just use simplified path, else use the Operating system path
            kind = (fileSystem->getOSPathKind() == OSPathKind::None) ? PathKind::Simplified
                                                                     : PathKind::OperatingSystem;
            return getPath(kind, path, outPath);
        }
    case PathKind::Canonical:
        {
            String canonicalPath;
            SLANG_RETURN_ON_FAIL(_getCanonicalPath(path, canonicalPath));
            *outPath = StringBlob::moveCreate(canonicalPath).detach();
            return SLANG_OK;
        }
    case PathKind::OperatingSystem:
        {
            String fixedPath;
            SLANG_RETURN_ON_FAIL(_getFixedPath(path, fixedPath));
            return fileSystem->getPath(kind, fixedPath.getBuffer(), outPath);
        }
    }

    return SLANG_FAIL;
}

void RelativeFileSystem::clearCache()
{
    auto fileSystem = _getExt();
    if (!fileSystem)
        return;

    fileSystem->clearCache();
}

SlangResult RelativeFileSystem::enumeratePathContents(
    const char* path,
    FileSystemContentsCallBack callback,
    void* userData)
{
    auto fileSystem = _getExt();
    if (!fileSystem)
        return SLANG_E_NOT_IMPLEMENTED;

    String fixedPath;
    SLANG_RETURN_ON_FAIL(_getFixedPath(path, fixedPath));
    return fileSystem->enumeratePathContents(fixedPath.getBuffer(), callback, userData);
}

SlangResult RelativeFileSystem::saveFile(const char* path, const void* data, size_t size)
{
    auto fileSystem = _getMutable();
    if (!fileSystem)
        return SLANG_E_NOT_IMPLEMENTED;

    String fixedPath;
    SLANG_RETURN_ON_FAIL(_getFixedPath(path, fixedPath));
    return fileSystem->saveFile(fixedPath.getBuffer(), data, size);
}

SlangResult RelativeFileSystem::saveFileBlob(const char* path, ISlangBlob* dataBlob)
{
    auto fileSystem = _getMutable();
    if (!fileSystem)
        return SLANG_E_NOT_IMPLEMENTED;

    String fixedPath;
    SLANG_RETURN_ON_FAIL(_getFixedPath(path, fixedPath));
    return fileSystem->saveFileBlob(fixedPath.getBuffer(), dataBlob);
}

SlangResult RelativeFileSystem::remove(const char* path)
{
    auto fileSystem = _getMutable();
    if (!fileSystem)
        return SLANG_E_NOT_IMPLEMENTED;

    String fixedPath;
    SLANG_RETURN_ON_FAIL(_getFixedPath(path, fixedPath));
    return fileSystem->remove(fixedPath.getBuffer());
}

SlangResult RelativeFileSystem::createDirectory(const char* path)
{
    auto fileSystem = _getMutable();
    if (!fileSystem)
        return SLANG_E_NOT_IMPLEMENTED;

    String fixedPath;
    SLANG_RETURN_ON_FAIL(_getFixedPath(path, fixedPath));
    return fileSystem->createDirectory(fixedPath.getBuffer());
}

} // namespace Slang
