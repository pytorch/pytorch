#ifndef SLANG_CORE_MEMORY_FILE_SYSTEM_H
#define SLANG_CORE_MEMORY_FILE_SYSTEM_H

#include "slang-basic.h"
#include "slang-com-object.h"
#include "slang-com-ptr.h"

namespace Slang
{

/* MemoryFileSystem is an implementation of ISlangMutableFileSystem that stores file contents in
'blobs' (typically) in memory.

A derived class can change how the contents of the contents blob is interpretted (so for example the
RiffFileSystem is implemented such that the Entry.m_contents is the files contents compressed).

The implementation uses a map to store the file/directory based on their canonical path. This makes
access relatively fast and simple - an access only requires a path being converted into a canonical
path, and then a lookup. Whilst this makes typical access fast, it means doing an enumeration of a
directory slower as it requires traversing all entries to find which are in the path.

This is in contrast with an implementation that held items in directories 'objects'. In that
scenario the path through the hierarchy would need to be traversed to find the item. Finding all of
the items in a directory is very fast - it's all the items held in the the directory 'object'.

TODO(JS):
* We may want to make saveFile take a blob, or have a version that does. Doing so would allow the
application to handle memory management around the blob.
*/
class MemoryFileSystem : public ISlangMutableFileSystem, public ComBaseObject
{
public:
    // ISlangUnknown
    SLANG_COM_BASE_IUNKNOWN_ALL

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
        return OSPathKind::None;
    }

    // ISlangModifyableFileSystem
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    saveFile(const char* path, const void* data, size_t size) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    saveFileBlob(const char* path, ISlangBlob* dataBlob) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL remove(const char* path) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL createDirectory(const char* path) SLANG_OVERRIDE;

    /// Ctor
    MemoryFileSystem();

protected:
    struct Entry
    {
        void reset()
        {
            m_type = SLANG_PATH_TYPE_FILE;
            m_canonicalPath = String();
            m_uncompressedSizeInBytes = 0;
            m_contents.setNull();
        }

        void initDirectory(const String& canonicalPath)
        {
            m_type = SLANG_PATH_TYPE_DIRECTORY;
            m_canonicalPath = canonicalPath;
            m_uncompressedSizeInBytes = 0;
            m_contents.setNull();
        }
        void initFile(const String& canonicalPath, size_t uncompressedSize, ISlangBlob* blob)
        {
            m_type = SLANG_PATH_TYPE_FILE;
            m_canonicalPath = canonicalPath;
            setContents(uncompressedSize, blob);
        }
        void initFile(const String& canonicalPath)
        {
            m_type = SLANG_PATH_TYPE_FILE;
            m_canonicalPath = canonicalPath;
            m_contents.setNull();
            m_uncompressedSizeInBytes = 0;
        }
        void setContents(size_t uncompressedSize, ISlangBlob* blob)
        {
            SLANG_ASSERT(m_type == SLANG_PATH_TYPE_FILE);
            SLANG_ASSERT(blob);
            m_uncompressedSizeInBytes = uncompressedSize;
            m_contents = blob;
        }

        SlangPathType m_type;
        String m_canonicalPath;

        /// The size as seen on the file system. Might be different from the size of m_contents
        /// if it's actually being stored in some other representation (such as compressed)
        size_t m_uncompressedSizeInBytes;
        ComPtr<ISlangBlob> m_contents; ///< Can be compressed or not
    };

    void* getInterface(const Guid& guid);
    void* getObject(const Guid& guid);

    Entry* _getEntryFromPath(const char* path, String* outPath = nullptr);
    Entry* _getEntryFromCanonicalPath(const String& canonicalPath);
    /// Creates or returns a file entry for the given path.
    /// If created the entry is empty.
    SlangResult _requireFile(const char* path, Entry** outEntry);
    /// Given the path returns the entry if it's a file, or returns an error
    SlangResult _loadFile(const char* path, Entry** outEntry);

    /// Given a path returns a canonical path.
    /// The canonical path must have *existing* parent paths.
    SlangResult _getCanonicalWithExistingParent(const char* path, StringBuilder& canonicalPath);

    /// Given a path returns a canonical path.
    SlangResult _getCanonical(const char* path, StringBuilder& canonicalPath);

    /// Clear, ensures any backing memory is also freed
    void _clear();

    // Maps canonical paths to an entries (which could be files or directories)
    Dictionary<String, Entry> m_entries;

    Entry m_rootEntry;
};

} // namespace Slang

#endif
