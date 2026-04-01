#include "slang-lz4-compression-system.h"

#include "slang-blob.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"

#include <lz4.h>

namespace Slang
{

// Allocate static const storage for the various interface IDs that the Slang API needs to expose

class LZ4CompressionSystemImpl : public RefObject, public ICompressionSystem
{
public:
    // ISlangUnknown
    // override ref counting, as singleton
    SLANG_IUNKNOWN_QUERY_INTERFACE
    SLANG_NO_THROW uint32_t SLANG_MCALL addRef() SLANG_OVERRIDE { return 1; }
    SLANG_NO_THROW uint32_t SLANG_MCALL release() SLANG_OVERRIDE { return 1; }

    // ICompressionSystem
    virtual SLANG_NO_THROW CompressionSystemType SLANG_MCALL getSystemType() SLANG_OVERRIDE
    {
        return CompressionSystemType::LZ4;
    }
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL compress(
        const CompressionStyle* style,
        const void* src,
        size_t srcSizeInBytes,
        ISlangBlob** outBlob) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL decompress(
        const void* compressed,
        size_t compressedSizeInBytes,
        size_t decompressedSizeInBytes,
        void* outDecompressed) SLANG_OVERRIDE;

protected:
    ICompressionSystem* getInterface(const Guid& guid);
};

ICompressionSystem* LZ4CompressionSystemImpl::getInterface(const Guid& guid)
{
    return (guid == ISlangUnknown::getTypeGuid() || guid == ICompressionSystem::getTypeGuid())
               ? static_cast<ICompressionSystem*>(this)
               : nullptr;
}

SlangResult LZ4CompressionSystemImpl::compress(
    const CompressionStyle* style,
    const void* src,
    size_t srcSizeInBytes,
    ISlangBlob** outBlob)
{
    SLANG_UNUSED(style);
    const size_t compressedBound = LZ4_compressBound(int(srcSizeInBytes));

    ScopedAllocation alloc;
    void* compressedData = alloc.allocate(compressedBound);

    const int compressedSize = LZ4_compress_default(
        (const char*)src,
        (char*)compressedData,
        int(srcSizeInBytes),
        int(compressedBound));
    alloc.reallocate(compressedSize);

    auto blob = RawBlob::moveCreate(alloc);

    *outBlob = blob.detach();
    return SLANG_OK;
}

SlangResult LZ4CompressionSystemImpl::decompress(
    const void* compressed,
    size_t compressedSizeInBytes,
    size_t decompressedSizeInBytes,
    void* outDecompressed)
{
    const int decompressedSize = LZ4_decompress_safe(
        (const char*)compressed,
        (char*)outDecompressed,
        int(compressedSizeInBytes),
        int(decompressedSizeInBytes));
    SLANG_UNUSED(decompressedSize);
    SLANG_ASSERT(size_t(decompressedSize) == decompressedSizeInBytes);
    return SLANG_OK;
}

/* static */ ICompressionSystem* LZ4CompressionSystem::getSingleton()
{
    static LZ4CompressionSystemImpl impl;
    return &impl;
}

} // namespace Slang
