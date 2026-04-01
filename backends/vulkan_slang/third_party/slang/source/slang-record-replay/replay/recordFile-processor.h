#ifndef FILE_PROCESSOR_H
#define FILE_PROCESSOR_H

#include "../../core/slang-stream.h"
#include "../util/record-utility.h"
#include "slang-decoder.h"

#include <cstdlib>

namespace SlangRecord
{

enum RecordFileResultCode
{
    RESULT_OK = 0x00,
    NOT_EXSIT = 0x01,
    ERROR_BLOCK = 0x02
};

class RecordFileProcessor
{
public:
    RecordFileProcessor(const Slang::String& filePath);

    bool addDecoder(SlangDecoder* pDecoder)
    {
        if (pDecoder == nullptr)
        {
            slangRecordLog(LogLevel::Error, "Decoder is nullptr\n");
            return false;
        }
        m_decoder = pDecoder;
        return true;
    }

    bool processNextBlock();
    bool processHeader(FunctionHeader& header);
    RecordFileResultCode processTailer(FunctionTailer& tailer);
    bool processMethod(FunctionHeader const& header, const uint8_t* buffer, int64_t bufferSize);
    bool processFunction(FunctionHeader const& header, const uint8_t* buffer, int64_t bufferSize);

private:
    Slang::FileStream m_inputStream;
    Slang::List<uint8_t> m_parameterBuffer;
    Slang::List<uint8_t> m_outputBuffer;

    SlangDecoder* m_decoder = nullptr;
};

} // namespace SlangRecord

#endif // FILE_PROCESSOR_H
