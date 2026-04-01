#include "slang-riff-file-system.h"

#include "slang-blob.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"

// Compression systems
#include "slang-deflate-compression-system.h"
#include "slang-lz4-compression-system.h"

namespace Slang
{

RiffFileSystem::RiffFileSystem(ICompressionSystem* compressionSystem)
    : m_compressionSystem(compressionSystem)
{
}

void* RiffFileSystem::getInterface(const Guid& guid)
{
    if (auto ptr = Super::getInterface(guid))
    {
        return ptr;
    }
    else if (guid == IArchiveFileSystem::getTypeGuid())
    {
        return static_cast<IArchiveFileSystem*>(this);
    }
    return nullptr;
}

void* RiffFileSystem::getObject(const Guid& guid)
{
    SLANG_UNUSED(guid);
    return nullptr;
}

void* RiffFileSystem::castAs(const Guid& guid)
{
    if (auto ptr = getInterface(guid))
    {
        return ptr;
    }
    return getObject(guid);
}

SlangResult RiffFileSystem::loadFile(char const* path, ISlangBlob** outBlob)
{
    Entry* entry;
    SLANG_RETURN_ON_FAIL(_loadFile(path, &entry));

    ISlangBlob* contents = entry->m_contents;

    if (m_compressionSystem)
    {
        // Okay lets decompress into a blob
        ScopedAllocation alloc;
        void* dst = alloc.allocateTerminated(entry->m_uncompressedSizeInBytes);
        SLANG_RETURN_ON_FAIL(m_compressionSystem->decompress(
            contents->getBufferPointer(),
            contents->getBufferSize(),
            entry->m_uncompressedSizeInBytes,
            dst));

        auto blob = RawBlob::moveCreate(alloc);

        *outBlob = blob.detach();
        return SLANG_OK;
    }
    else
    {
        // Just return as is
        contents->addRef();
        *outBlob = contents;
        return SLANG_OK;
    }
}

SlangResult RiffFileSystem::saveFile(const char* path, const void* data, size_t size)
{
    Entry* entry;
    SLANG_RETURN_ON_FAIL(_requireFile(path, &entry));

    ComPtr<ISlangBlob> contents;
    if (m_compressionSystem)
    {
        // Lets try compressing the input
        SLANG_RETURN_ON_FAIL(
            m_compressionSystem->compress(&m_compressionStyle, data, size, contents.writeRef()));
    }
    else
    {
        // Just store the data directly.
        contents = RawBlob::create(data, size);
    }
    entry->setContents(size, contents);
    return SLANG_OK;
}

SlangResult RiffFileSystem::saveFileBlob(const char* path, ISlangBlob* dataBlob)
{
    if (!dataBlob)
    {
        return SLANG_E_INVALID_ARG;
    }

    if (m_compressionSystem)
    {
        return saveFile(path, dataBlob->getBufferPointer(), dataBlob->getBufferSize());
    }
    else
    {
        return Super::saveFileBlob(path, dataBlob);
    }
}

SlangResult RiffFileSystem::loadArchive(const void* archive, size_t archiveSizeInBytes)
{
    // Load the riff
    RiffContainer container;

    MemoryStreamBase stream(FileAccess::Read, archive, archiveSizeInBytes);
    SLANG_RETURN_ON_FAIL(RiffUtil::read(&stream, container));

    RiffContainer::ListChunk* rootList = container.getRoot();
    // Make sure it's the right type
    if (rootList == nullptr || rootList->m_fourCC != RiffFileSystemBinary::kContainerFourCC)
    {
        return SLANG_FAIL;
    }

    // Clear the contents
    _clear();

    // Find the header
    const auto header = rootList->findContainedData<RiffFileSystemBinary::Header>(
        RiffFileSystemBinary::kHeaderFourCC);

    CompressionSystemType compressionType = CompressionSystemType(header->compressionSystemType);
    switch (compressionType)
    {
    case CompressionSystemType::None:
        {
            // Null m_compressionSystem means no compression
            m_compressionSystem.setNull();
            break;
        }
    case CompressionSystemType::Deflate:
        {
            m_compressionSystem = DeflateCompressionSystem::getSingleton();
            break;
        }
    case CompressionSystemType::LZ4:
        {
            m_compressionSystem = LZ4CompressionSystem::getSingleton();
            break;
        }
    default:
        return SLANG_FAIL;
    }

    // Read all of the contained data

    {
        List<RiffContainer::DataChunk*> srcEntries;
        rootList->findContained(RiffFileSystemBinary::kEntryFourCC, srcEntries);

        for (auto chunk : srcEntries)
        {
            auto data = chunk->getSingleData();

            const uint8_t* srcData = (const uint8_t*)data->getPayload();
            const size_t dataSize = data->getSize();

            if (dataSize < sizeof(RiffFileSystemBinary::Entry))
            {
                return SLANG_FAIL;
            }

            auto srcEntry = (const RiffFileSystemBinary::Entry*)srcData;
            srcData += sizeof(*srcEntry);

            // Check if seems plausible
            if (sizeof(RiffFileSystemBinary::Entry) + srcEntry->compressedSize +
                    srcEntry->pathSize !=
                dataSize)
            {
                return SLANG_FAIL;
            }

            Entry dstEntry;

            const char* path = (const char*)srcData;
            srcData += srcEntry->pathSize;

            dstEntry.m_canonicalPath = UnownedStringSlice(path, srcEntry->pathSize - 1);
            dstEntry.m_type = (SlangPathType)srcEntry->pathType;
            dstEntry.m_uncompressedSizeInBytes = srcEntry->uncompressedSize;

            switch (dstEntry.m_type)
            {
            case SLANG_PATH_TYPE_FILE:
                {
                    if (srcData + srcEntry->compressedSize != data->getPayloadEnd())
                    {
                        return SLANG_FAIL;
                    }

                    // Get the compressed data
                    dstEntry.m_contents = RawBlob::create(srcData, srcEntry->compressedSize);
                    break;
                }
            case SLANG_PATH_TYPE_DIRECTORY:
                break;
            default:
                return SLANG_FAIL;
            }

            // If it's the root entry we can ignore (as already added)
            if (dstEntry.m_canonicalPath == ".")
            {
                continue;
            }

            // Add to the list of entries
            m_entries.add(dstEntry.m_canonicalPath, dstEntry);
        }
    }

    return SLANG_OK;
}

SlangResult RiffFileSystem::storeArchive(bool blobOwnsContent, ISlangBlob** outBlob)
{
    // All blobs are owned in this style
    SLANG_UNUSED(blobOwnsContent)

    RiffContainer container;
    RiffContainer::ScopeChunk scopeContainer(
        &container,
        RiffContainer::Chunk::Kind::List,
        RiffFileSystemBinary::kContainerFourCC);

    {
        RiffFileSystemBinary::Header header;
        CompressionSystemType compressionSystemType = m_compressionSystem
                                                          ? m_compressionSystem->getSystemType()
                                                          : CompressionSystemType::None;
        header.compressionSystemType = uint32_t(compressionSystemType);
        container.addDataChunk(RiffFileSystemBinary::kHeaderFourCC, &header, sizeof(header));
    }

    for (const auto& [_, srcEntry] : m_entries)
    {
        // Ignore the root entry
        if (srcEntry.m_canonicalPath == toSlice("."))
        {
            continue;
        }

        RiffContainer::ScopeChunk scopeData(
            &container,
            RiffContainer::Chunk::Kind::Data,
            RiffFileSystemBinary::kEntryFourCC);

        RiffFileSystemBinary::Entry dstEntry;
        dstEntry.uncompressedSize = 0;
        dstEntry.compressedSize = 0;
        dstEntry.pathSize = uint32_t(srcEntry.m_canonicalPath.getLength() + 1);
        dstEntry.pathType = srcEntry.m_type;

        ISlangBlob* blob = srcEntry.m_contents;

        if (srcEntry.m_type == SLANG_PATH_TYPE_FILE)
        {
            dstEntry.compressedSize = uint32_t(blob->getBufferSize());
            dstEntry.uncompressedSize = uint32_t(srcEntry.m_uncompressedSizeInBytes);
        }

        // Entry header
        container.write(&dstEntry, sizeof(dstEntry));

        // Path
        container.write(
            srcEntry.m_canonicalPath.getBuffer(),
            srcEntry.m_canonicalPath.getLength() + 1);

        // Add the contained data without copying
        if (blob)
        {
            RiffContainer::Data* data = container.addData();
            container.setUnowned(
                data,
                const_cast<void*>(blob->getBufferPointer()),
                blob->getBufferSize());
        }
    }

    OwnedMemoryStream stream(FileAccess::Write);
    // We now write the RiffContainer to the stream
    SLANG_RETURN_ON_FAIL(RiffUtil::write(container.getRoot(), true, &stream));

    List<uint8_t> data;
    stream.swapContents(data);

    *outBlob = ListBlob::moveCreate(data).detach();
    return SLANG_OK;
}

/* static */ bool RiffFileSystem::isArchive(const void* data, size_t sizeInBytes)
{
    MemoryStreamBase stream(FileAccess::Read, data, sizeInBytes);
    RiffListHeader header;
    return SLANG_SUCCEEDED(RiffUtil::readHeader(&stream, header)) &&
           header.subType == RiffFileSystemBinary::kContainerFourCC;
}

} // namespace Slang
