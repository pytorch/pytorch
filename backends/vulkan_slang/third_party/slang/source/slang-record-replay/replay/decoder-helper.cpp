#include "decoder-helper.h"

#include "parameter-decoder.h"

#include <cstdlib>
#include <vector>

namespace SlangRecord
{
DecoderAllocatorSingleton* DecoderAllocatorSingleton::getInstance()
{
    thread_local DecoderAllocatorSingleton instance;
    return &instance;
}

void* DecoderAllocatorSingleton::allocate(size_t size)
{
    void* data = calloc(1, size);

    if (!data)
    {
        slangRecordLog(LogLevel::Error, "Failed to allocate memory\n");
        std::abort();
    }

    m_allocations.add(data);
    return data;
}

DecoderAllocatorSingleton::~DecoderAllocatorSingleton()
{
    for (auto allocation : m_allocations)
    {
        free(allocation);
    }
}

template<typename T, typename U>
size_t StructDecoder<T, U>::decode(const uint8_t* buffer, int64_t bufferSize)
{
    return ParameterDecoder::decodeStruct(buffer, bufferSize, *this);
}

size_t BlobDecoder::decode(const uint8_t* buffer, int64_t bufferSize)
{
    size_t readByte = 0;
    readByte = ParameterDecoder::decodeAddress(buffer, bufferSize, m_address);

    if (!m_address)
    {
        readByte +=
            ParameterDecoder::decodePointer(buffer + readByte, bufferSize - readByte, m_blobData);
    }
    return readByte;
}

template class StructDecoder<SlangGlobalSessionDesc>;
template class StructDecoder<slang::SessionDesc>;
template class StructDecoder<slang::PreprocessorMacroDesc>;
template class StructDecoder<slang::CompilerOptionEntry>;
template class StructDecoder<slang::CompilerOptionValue>;
template class StructDecoder<slang::TargetDesc>;
template class StructDecoder<slang::SpecializationArg>;
} // namespace SlangRecord
