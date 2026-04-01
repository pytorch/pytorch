#include "slang-deflate-compression-system.h"

#include "slang-com-helper.h"
#include "slang-com-ptr.h"

// We don't want compress #define to clash
#define MINIZ_NO_ZLIB_COMPATIBLE_NAMES 1

#include "slang-blob.h"

#include <miniz.h>

namespace Slang
{

// Allocate static const storage for the various interface IDs that the Slang API needs to expose

class DeflateCompressionSystemImpl : public RefObject, public ICompressionSystem
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
        return CompressionSystemType::Deflate;
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

ICompressionSystem* DeflateCompressionSystemImpl::getInterface(const Guid& guid)
{
    return (guid == ISlangUnknown::getTypeGuid() || guid == ICompressionSystem::getTypeGuid())
               ? static_cast<ICompressionSystem*>(this)
               : nullptr;
}

SlangResult DeflateCompressionSystemImpl::compress(
    const CompressionStyle* style,
    const void* src,
    size_t srcSizeInBytes,
    ISlangBlob** outBlob)
{
    SLANG_UNUSED(style);

    size_t compressedSizeInBytes;

    const int flags = 0;
    void* compressed =
        tdefl_compress_mem_to_heap(src, srcSizeInBytes, &compressedSizeInBytes, flags);

    if (!compressed)
    {
        return SLANG_FAIL;
    }

    ScopedAllocation alloc;
    alloc.attach(compressed, compressedSizeInBytes);

    auto blob = RawBlob::moveCreate(alloc);
    *outBlob = blob.detach();
    return SLANG_OK;
}

SlangResult DeflateCompressionSystemImpl::decompress(
    const void* compressed,
    size_t compressedSizeInBytes,
    size_t decompressedSizeInBytes,
    void* outDecompressed)
{
    const int flags = TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF;

    size_t size = tinfl_decompress_mem_to_mem(
        outDecompressed,
        decompressedSizeInBytes,
        compressed,
        compressedSizeInBytes,
        flags);
    if (size == TINFL_DECOMPRESS_MEM_TO_MEM_FAILED)
    {
        return SLANG_FAIL;
    }

    return SLANG_OK;
}

/* static */ ICompressionSystem* DeflateCompressionSystem::getSingleton()
{
    static DeflateCompressionSystemImpl impl;
    return &impl;
}

} // namespace Slang
