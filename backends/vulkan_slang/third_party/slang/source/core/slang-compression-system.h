#ifndef SLANG_COMPRESSION_SYSTEM_H
#define SLANG_COMPRESSION_SYSTEM_H

#include "slang-basic.h"

namespace Slang
{

struct CompressionStyle
{
    enum class Type
    {
        Level,           ///< Use the value specified in 'level' to control compression
        BestSpeed,       ///< Best for speed (typically lower compression ration)
        BestCompression, ///< Best compression (typically slower)
        Default,         ///< Default compression (a good balance between speed and size)
    };
    Type m_type = Type::Default; ///< The type
    float m_level =
        1.0f; ///< 0 lowest compression, 1 highest compression (Ignored if m_type != Type::Level)
};

enum class CompressionSystemType
{
    None,
    Deflate,
    LZ4,
    CountOf,
};

class ICompressionSystem : public ISlangUnknown
{
    SLANG_COM_INTERFACE(
        0xcc935840,
        0xe059,
        0x4bb8,
        {0xa2, 0x2d, 0x92, 0x7b, 0x3c, 0x73, 0x8f, 0x85})

    /** Get the compression system type
    @return The compression system type */
    virtual SLANG_NO_THROW CompressionSystemType SLANG_MCALL getSystemType() = 0;

    /** compress
    @param src Points to the start of the data to compress
    @param srcSizeInBytes The size of the source data to compress in bytes
    @param outBlob The input data compressed
    @return SLANG_OK if successful */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL compress(
        const CompressionStyle* style,
        const void* src,
        size_t srcSizeInBytes,
        ISlangBlob** outBlob) = 0;

    /* decompress
    @param compressed The start of the compressed data
    @param compressedSizeInBytes The compressed size in bytes
    @param decompressedSizeInBytes The size of the decompressed buffer. MUST be exactly the same as
    the original source size.
    @param outDecompressed Where decompressed data is written
    @return SLANG_OK if successful */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL decompress(
        const void* compressed,
        size_t compressedSizeInBytes,
        size_t decompressedSizeInBytes,
        void* outDecompressed) = 0;
};

} // namespace Slang

#endif
