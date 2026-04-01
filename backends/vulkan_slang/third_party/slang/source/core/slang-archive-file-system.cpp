#include "slang-archive-file-system.h"

#include "../core/slang-castable.h"
#include "slang-blob.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang-io.h"
#include "slang-riff-file-system.h"
#include "slang-string-util.h"

// Compression systems
#include "slang-deflate-compression-system.h"
#include "slang-lz4-compression-system.h"

// Zip file system
#include "slang-riff.h"
#include "slang-zip-file-system.h"

namespace Slang
{

SlangResult loadArchiveFileSystem(
    const void* data,
    size_t dataSizeInBytes,
    ComPtr<ISlangFileSystemExt>& outFileSystem)
{
    ComPtr<ISlangMutableFileSystem> fileSystem;
    if (ZipFileSystem::isArchive(data, dataSizeInBytes))
    {
        // It's a zip
        SLANG_RETURN_ON_FAIL(ZipFileSystem::create(fileSystem));
    }
    else if (RiffFileSystem::isArchive(data, dataSizeInBytes))
    {
        // It's riff contained (Slang specific)
        fileSystem = new RiffFileSystem(nullptr);
    }
    else
    {
        return SLANG_FAIL;
    }

    auto archiveFileSystem = as<IArchiveFileSystem>(fileSystem);
    if (!archiveFileSystem)
    {
        return SLANG_FAIL;
    }

    SLANG_RETURN_ON_FAIL(archiveFileSystem->loadArchive(data, dataSizeInBytes));

    outFileSystem = fileSystem;
    return SLANG_OK;
}

SlangResult createArchiveFileSystem(
    SlangArchiveType type,
    ComPtr<ISlangMutableFileSystem>& outFileSystem)
{
    switch (type)
    {
    case SLANG_ARCHIVE_TYPE_ZIP:
        {
            return ZipFileSystem::create(outFileSystem);
        }
    case SLANG_ARCHIVE_TYPE_RIFF:
        {
            outFileSystem = new RiffFileSystem(nullptr);
            return SLANG_OK;
        }
    case SLANG_ARCHIVE_TYPE_RIFF_DEFLATE:
        {
            outFileSystem = new RiffFileSystem(DeflateCompressionSystem::getSingleton());
            return SLANG_OK;
        }
    case SLANG_ARCHIVE_TYPE_RIFF_LZ4:
        {
            outFileSystem = new RiffFileSystem(LZ4CompressionSystem::getSingleton());
            return SLANG_OK;
        }
    }

    return SLANG_FAIL;
}

} // namespace Slang
