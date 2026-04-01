#ifndef SLANG_FILE_SYSTEM_H_INCLUDED
#define SLANG_FILE_SYSTEM_H_INCLUDED

#include "../core/slang-blob.h"
#include "../core/slang-dictionary.h"
#include "../core/slang-string-util.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang.h"

namespace Slang
{

enum class FileSystemStyle
{
    Load,    ///< Equivalent to ISlangFileSystem
    Ext,     ///< Equivalent to ISlangFileSystemExt
    Mutable, ///< Equivalent to ISlangModifyableFileSystem
};

// Can be used for all styles of file system
class OSFileSystem : public ISlangMutableFileSystem
{
public:
    // ISlangUnknown
    // override ref counting, as DefaultFileSystem is singleton
    SLANG_IUNKNOWN_QUERY_INTERFACE
    SLANG_NO_THROW uint32_t SLANG_MCALL addRef() SLANG_OVERRIDE { return 1; }
    SLANG_NO_THROW uint32_t SLANG_MCALL release() SLANG_OVERRIDE { return 1; }

    // ISlangCastable
    virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const Guid& guid) SLANG_OVERRIDE;

    // ISlangFileSystem
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL loadFile(char const* path, ISlangBlob** outBlob)
        SLANG_OVERRIDE;

    // ISlangFileSystemExt
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getFileUniqueIdentity(const char* path, ISlangBlob** uniqueIdentityOut) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL calcCombinedPath(
        SlangPathType fromPathType,
        const char* fromPath,
        const char* path,
        ISlangBlob** pathOut) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getPathType(const char* path, SlangPathType* pathTypeOut) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getPath(PathKind pathKind, const char* path, ISlangBlob** outPath) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL clearCache() SLANG_OVERRIDE {}
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL enumeratePathContents(
        const char* path,
        FileSystemContentsCallBack callback,
        void* userData) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW OSPathKind SLANG_MCALL getOSPathKind() SLANG_OVERRIDE
    {
        return OSPathKind::Direct;
    }

    // ISlangModifyableFileSystem
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    saveFile(const char* path, const void* data, size_t size) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    saveFileBlob(const char* path, ISlangBlob* dataBlob) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL remove(const char* path) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL createDirectory(const char* path) SLANG_OVERRIDE;

    /// Get a default instance
    static ISlangFileSystem* getLoadSingleton() { return &g_load; }
    static ISlangFileSystemExt* getExtSingleton() { return &g_ext; }
    static ISlangMutableFileSystem* getMutableSingleton() { return &g_mutable; }

private:
    /// Make so not constructible
    OSFileSystem(FileSystemStyle style)
        : m_style(style)
    {
    }

    virtual ~OSFileSystem() {}

    ISlangUnknown* getInterface(const Guid& guid);
    void* getObject(const Guid& guid);

    FileSystemStyle m_style;

    static OSFileSystem g_load;
    static OSFileSystem g_ext;
    static OSFileSystem g_mutable;
};

/* Wraps an underlying ISlangFileSystem or ISlangFileSystemExt and provides caching,
as well as emulation of methods if only has ISlangFileSystem interface. Will query capabilities
of the interface on the constructor.

NOTE! That this behavior is the same as previously in that....
1) calcRelativePath, just returns the path as processed by the Path:: methods
2) getUniqueIdentity behavior depends on the UniqueIdentityMode.
*/
class CacheFileSystem : public ISlangFileSystemExt, public ComBaseObject
{
public:
    SLANG_CLASS_GUID(0x2f4d1d03, 0xa0d1, 0x434b, {0x87, 0x7a, 0x65, 0x5, 0xa4, 0xa0, 0x9a, 0x3b})

    enum class PathStyle
    {
        Default,       ///< Pass to say use the default
        Simplifiable,  ///< It can be simplified by Path::Simplify
        FileSystemExt, ///< Use file system
    };

    enum UniqueIdentityMode
    {
        Default,             ///< If passed, will default to the others depending on what kind of
                             ///< ISlangFileSystem is passed in
        Path,                ///< Just use the path as is (old style slang behavior)
        SimplifyPath,        ///< Use the input path 'simplified' (ie removing . and .. aspects)
        Hash,                ///< Use hashing
        SimplifyPathAndHash, ///< Tries simplifying path first, and if that doesn't work it hashes
        FileSystemExt,       ///< Use the file system extended interface.
    };

    /* Cannot change order/add members without changing s_compressedResultToResult */
    enum class CompressedResult : uint8_t
    {
        Uninitialized, ///< Holds no value
        Ok,            ///< Ok
        NotFound,      ///< File not found
        CannotOpen,    ///< Cannot open
        Fail,          ///< Generic failure
        CountOf,
    };

    struct PathInfo
    {
        PathInfo(const String& uniqueIdentity)
            : m_uniqueIdentity(uniqueIdentity)
        {
            m_loadFileResult = CompressedResult::Uninitialized;
            m_getPathTypeResult = CompressedResult::Uninitialized;
            m_getCanonicalPathResult = CompressedResult::Uninitialized;

            m_pathType = SLANG_PATH_TYPE_FILE;
        }

        /// Get the unique identity path as a string
        const String& getUniqueIdentity() const { return m_uniqueIdentity; }

        String m_uniqueIdentity;
        CompressedResult m_loadFileResult;
        CompressedResult m_getPathTypeResult;
        CompressedResult m_getCanonicalPathResult;

        SlangPathType m_pathType;
        ComPtr<ISlangBlob> m_fileBlob;
        String m_canonicalPath;
    };

    Dictionary<String, PathInfo*>& getPathMap() { return m_pathMap; }
    Dictionary<String, PathInfo*>& getUniqueMap() { return m_uniqueIdentityMap; }

    // ISlangUnknown
    SLANG_COM_BASE_IUNKNOWN_ALL

    // ISlangCastable
    virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const Guid& guid) SLANG_OVERRIDE;

    // ISlangFileSystem
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL loadFile(char const* path, ISlangBlob** outBlob)
        SLANG_OVERRIDE;

    // ISlangFileSystemExt
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getFileUniqueIdentity(const char* path, ISlangBlob** outUniqueIdentity) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL calcCombinedPath(
        SlangPathType fromPathType,
        const char* fromPath,
        const char* path,
        ISlangBlob** pathOut) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getPathType(const char* path, SlangPathType* outPathType) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getPath(PathKind kind, const char* path, ISlangBlob** outPath) SLANG_OVERRIDE;

    virtual SLANG_NO_THROW void SLANG_MCALL clearCache() SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL enumeratePathContents(
        const char* path,
        FileSystemContentsCallBack callback,
        void* userData) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW OSPathKind SLANG_MCALL getOSPathKind() SLANG_OVERRIDE
    {
        return m_osPathKind;
    }

    /// Get the unique identity mode
    UniqueIdentityMode getUniqueIdentityMode() const { return m_uniqueIdentityMode; }
    /// Get the path style
    PathStyle getPathStyle() const { return m_pathStyle; }

    /// Set the inner file system
    void setInnerFileSystem(
        ISlangFileSystem* fileSystem,
        UniqueIdentityMode uniqueIdentityMode = UniqueIdentityMode::Default,
        PathStyle pathStyle = PathStyle::Default);

    /// Ctor
    explicit CacheFileSystem(
        ISlangFileSystem* fileSystem,
        UniqueIdentityMode uniqueIdentityMode = UniqueIdentityMode::Default,
        PathStyle pathStyle = PathStyle::Default);
    /// Dtor
    virtual ~CacheFileSystem();

    static CompressedResult toCompressedResult(Result res);
    static Result toResult(CompressedResult compRes)
    {
        return s_compressedResultToResult[int(compRes)];
    }
    static const Result s_compressedResultToResult[int(CompressedResult::CountOf)];

protected:
    void* getInterface(const Guid& guid);
    void* getObject(const Guid& guid);

    SlangResult _getSimplifiedPath(const char* path, ISlangBlob** outSimplifiedPath);
    SlangResult _getCanonicalPath(const char* path, ISlangBlob** outCanonicalPath);

    /// Given a path, works out a uniqueIdentity, based on the uniqueIdentityMode.
    /// outFileContents will be set if file had to be read to produce the uniqueIdentity (ie with
    /// Hash) If the file doesn't have to be read, then outFileContents will be nullptr, even if it
    /// is backed by a file.
    SlangResult _calcUniqueIdentity(
        const String& path,
        String& outUniqueIdentity,
        ComPtr<ISlangBlob>& outFileContents);

    /// For a given path gets a PathInfo. Can return nullptr, if it is not possible to create the
    /// PathInfo for some reason
    PathInfo* _resolvePathCacheInfo(const String& path);
    /// Turns the path into a uniqueIdentity, and then tries to look up in the uniqueIdentityMap.
    PathInfo* _resolveUniqueIdentityCacheInfo(const String& path);
    /// Will simplify the path (if possible) to lookup on the pathCache else will create on
    /// uniqueIdentityMap
    PathInfo* _resolveSimplifiedPathCacheInfo(const String& path);

    SlangResult _getPathType(PathInfo* pathInfo, const char* inPath, SlangPathType* pathTypeOut);

    /* TODO: This may be improved by mapping to a ISlangBlob. This makes output fast and easy, and
    if constructed as a StringBlob, we can just static_cast to get as a string to use internally,
    instead of constantly converting. It is probably the case we cannot do dynamic_cast on
    ISlangBlob if we don't know where constructed -> if outside of slang codebase doing such a cast
    can cause an exception. So we *never* want to do dynamic cast from blobs which could be created
    by external code. */

    Dictionary<String, PathInfo*> m_pathMap; ///< Maps a path to a PathInfo (and unique identity)
    Dictionary<String, PathInfo*> m_uniqueIdentityMap; ///< Maps a unique identity for a file to its
                                                       ///< contents. This OWNs the PathInfo.

    UniqueIdentityMode m_uniqueIdentityMode; ///< Determines how the 'uniqueIdentity' is produced.
                                             ///< Cannot be Default in usage.
    PathStyle m_pathStyle;                   ///< Style of paths

    ComPtr<ISlangFileSystem> m_fileSystem; ///< Must always be set
    ComPtr<ISlangFileSystemExt>
        m_fileSystemExt; ///< Optionally set -> if nullptr will fall back on the m_fileSystem and
                         ///< emulate all the other methods of ISlangFileSystemExt

    OSPathKind m_osPathKind = OSPathKind::None; ///< OS path kind
};

class RelativeFileSystem : public ISlangMutableFileSystem, public ComBaseObject
{
public:
    SLANG_COM_BASE_IUNKNOWN_ALL

    // ISlangFileSystem
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL loadFile(char const* path, ISlangBlob** outBlob)
        SLANG_OVERRIDE;

    // ISlangCastable
    virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const Guid& guid) SLANG_OVERRIDE;

    // ISlangFileSystemExt
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getFileUniqueIdentity(const char* path, ISlangBlob** outUniqueIdentity) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL calcCombinedPath(
        SlangPathType fromPathType,
        const char* fromPath,
        const char* path,
        ISlangBlob** pathOut) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getPathType(const char* path, SlangPathType* outPathType) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getPath(PathKind pathKind, const char* path, ISlangBlob** outPath) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL clearCache() SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL enumeratePathContents(
        const char* path,
        FileSystemContentsCallBack callback,
        void* userData) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW OSPathKind SLANG_MCALL getOSPathKind() SLANG_OVERRIDE
    {
        return m_osPathKind;
    }

    // ISlangModifyableFileSystem
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    saveFile(const char* path, const void* data, size_t size) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    saveFileBlob(const char* path, ISlangBlob* dataBlob) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL remove(const char* path) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL createDirectory(const char* path) SLANG_OVERRIDE;

    /// stripPath will remove any path for an access, making an access always just
    /// access the *filename* from the input path, in the contained filesystem at the relative path
    RelativeFileSystem(
        ISlangFileSystem* fileSystem,
        const String& relativePath,
        bool stripPath = false);

protected:
    ISlangFileSystemExt* _getExt()
    {
        return Index(m_style) >= Index(FileSystemStyle::Ext)
                   ? reinterpret_cast<ISlangFileSystemExt*>(m_fileSystem.get())
                   : nullptr;
    }
    ISlangMutableFileSystem* _getMutable()
    {
        return Index(m_style) >= Index(FileSystemStyle::Mutable)
                   ? reinterpret_cast<ISlangMutableFileSystem*>(m_fileSystem.get())
                   : nullptr;
    }

    SlangResult _calcCombinedPathInner(
        SlangPathType fromPathType,
        const char* fromPath,
        const char* path,
        ISlangBlob** pathOut);

    /// Get the fixed path to the item for the backing file system.
    SlangResult _getFixedPath(const char* path, String& outPath);

    SlangResult _getCanonicalPath(const char* path, String& outPath);

    ISlangUnknown* getInterface(const Guid& guid);
    void* getObject(const Guid& guid);

    bool m_stripPath; ///< If set any path prior to an item will be stripped (making the directory
                      ///< in effect flat)

    FileSystemStyle m_style;
    ComPtr<ISlangFileSystem> m_fileSystem; ///< NOTE! Has to match what's in style, such style can
                                           ///< be reached via reinterpret_cast

    String m_relativePath;
    OSPathKind m_osPathKind = OSPathKind::None; ///< OS path kind
};

} // namespace Slang

#endif // SLANG_FILE_SYSTEM_H_INCLUDED
