#include "recordFile-processor.h"

#include "../util/record-format.h"
#include "parameter-decoder.h"

namespace SlangRecord
{
RecordFileProcessor::RecordFileProcessor(const Slang::String& filePath)
{
    Slang::FileMode fileMode = Slang::FileMode::Open;
    Slang::FileAccess fileAccess = Slang::FileAccess::Read;
    Slang::FileShare fileShare = Slang::FileShare::None;

    // Open the record file with read-only access
    SlangResult res = m_inputStream.init(filePath, fileMode, fileAccess, fileShare);

    if (res != SLANG_OK)
    {
        SlangRecord::slangRecordLog(
            SlangRecord::LogLevel::Error,
            "Failed to open file %s\n",
            filePath.begin());
        std::abort();
    }

    // Enable log system
    setLogLevel();
}

bool RecordFileProcessor::processNextBlock()
{
    FunctionHeader header{};
    if (!processHeader(header))
    {
        return false;
    }

    ApiClassId classId = static_cast<ApiClassId>(getClassId(header.callId));

    // capacity comparison will be performed in the reserve call, so we can safely call reserve
    m_parameterBuffer.reserve(header.dataSizeInBytes);

    size_t readBytes = 0;
    SlangResult res = SLANG_OK;

    if (header.dataSizeInBytes)
    {
        res = m_inputStream.read(m_parameterBuffer.getBuffer(), header.dataSizeInBytes, readBytes);
    }

    if (res != SLANG_OK || readBytes != header.dataSizeInBytes)
    {
        return false;
    }

    FunctionTailer tailer{};
    if (processTailer(tailer) == ERROR_BLOCK)
    {
        return false;
    }

    if (tailer.dataSizeInBytes)
    {
        m_outputBuffer.reserve(tailer.dataSizeInBytes);
        res = m_inputStream.read(m_outputBuffer.getBuffer(), tailer.dataSizeInBytes, readBytes);

        if (res != SLANG_OK || readBytes != tailer.dataSizeInBytes)
        {
            return false;
        }
    }

    bool ret = false;
    SlangDecoder::ParameterBlock paramBlock{};
    paramBlock.parameterBuffer = m_parameterBuffer.getBuffer();
    paramBlock.parameterBufferSize = header.dataSizeInBytes;
    paramBlock.outputBuffer = m_outputBuffer.getBuffer();
    paramBlock.outputBufferSize = tailer.dataSizeInBytes;

    if (classId == ApiClassId::GlobalFunction)
    {
        ret = m_decoder->processFunctionCall(header, paramBlock);
    }
    else
    {
        ret = m_decoder->processMethodCall(header, paramBlock);
    }

    m_parameterBuffer.clear();
    m_outputBuffer.clear();
    return ret;
}

bool RecordFileProcessor::processHeader(FunctionHeader& header)
{
    size_t readBytes = 0;
    SlangResult res = m_inputStream.read(&header, sizeof(FunctionHeader), readBytes);

    if (res != SLANG_OK || readBytes != sizeof(FunctionHeader))
    {
        return false;
    }

    if (header.magic != MAGIC_HEADER || header.callId == ApiCallId::InvalidCallId)
    {
        return false;
    }

    return true;
}

RecordFileResultCode RecordFileProcessor::processTailer(FunctionTailer& tailer)
{
    size_t readBytes = 0;
    SlangResult res = m_inputStream.read(&tailer, sizeof(FunctionTailer), readBytes);

    if (res != SLANG_OK || readBytes != sizeof(FunctionTailer))
    {
        return ERROR_BLOCK;
    }

    // If we don't read a valid tailer, but the magic is bit is a header, it indicates that
    // there is no tailer for this block, but it's still a valid block.
    if (tailer.magic == MAGIC_HEADER)
    {
        // revert back to last read position, and clear tailer
        int64_t offset = -(int64_t)sizeof(FunctionTailer);
        m_inputStream.seek(Slang::SeekOrigin::Current, offset);
        memset(&tailer, 0, sizeof(FunctionTailer));
        return NOT_EXSIT;
    }

    if (tailer.magic != MAGIC_TAILER)
    {
        return ERROR_BLOCK;
    }

    return RESULT_OK;
}

bool RecordFileProcessor::processMethod(
    FunctionHeader const& header,
    const uint8_t* parameterBuffer,
    int64_t bufferSize)
{
    return false;
}

bool RecordFileProcessor::processFunction(
    FunctionHeader const& header,
    const uint8_t* parameterBuffer,
    int64_t bufferSize)
{
    return false;
}
}; // namespace SlangRecord
