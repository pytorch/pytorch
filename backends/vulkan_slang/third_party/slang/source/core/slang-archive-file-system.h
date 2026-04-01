#ifndef SLANG_CORE_ARCHIVE_FILE_SYSTEM_H
#define SLANG_CORE_ARCHIVE_FILE_SYSTEM_H

#include "slang-basic.h"
#include "slang-com-ptr.h"
#include "slang-compression-system.h"

namespace Slang
{

class IArchiveFileSystem : public ISlangCastable
{
    SLANG_COM_INTERFACE(
        0x5c565aac,
        0xe834,
        0x41fc,
        {0x8b, 0xb, 0x7d, 0x4c, 0xf3, 0x8b, 0x89, 0x50});

    /// Loads an archive.
    SLANG_NO_THROW virtual SlangResult SLANG_MCALL
    loadArchive(const void* archive, size_t archiveSizeInBytes) = 0;
    /// Get as an archive (that can be saved to disk)
    /// NOTE! If the blob is not owned, it's contents can be invalidated by any call to a method of
    /// the file system or loss of scope
    SLANG_NO_THROW virtual SlangResult SLANG_MCALL
    storeArchive(bool blobOwnsContent, ISlangBlob** outBlob) = 0;
    /// Set the compression - used for any subsequent items added
    SLANG_NO_THROW virtual void SLANG_MCALL setCompressionStyle(const CompressionStyle& style) = 0;
};

SlangResult loadArchiveFileSystem(
    const void* data,
    size_t dataSizeInBytes,
    ComPtr<ISlangFileSystemExt>& outFileSystem);
SlangResult createArchiveFileSystem(
    SlangArchiveType type,
    ComPtr<ISlangMutableFileSystem>& outFileSystem);

} // namespace Slang

#endif
