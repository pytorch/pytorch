#ifndef RECORD_MANAGER_H
#define RECORD_MANAGER_H

#include "../../core/slang-io.h"
#include "../../core/slang-string.h"
#include "../util/record-format.h"
#include "parameter-recorder.h"

namespace SlangRecord
{
class RecordManager : public Slang::RefObject
{
public:
    RecordManager(uint64_t globalSessionHandle);

    // Each method record has to start with a FunctionHeader
    ParameterRecorder* beginMethodRecord(const ApiCallId& callId, uint64_t handleId);
    ParameterRecorder* endMethodRecord();

    // apendOutput is an optional call that can be used to append output to
    // the end of the record. It has to start with a FunctionTailer
    void apendOutput();

    const Slang::String& getRecordFileDirectory() const { return m_recordFileDirectory; }

private:
    void clearWithHeader(const ApiCallId& callId, uint64_t handleId);
    void clearWithTailer();

    MemoryStream m_memoryStream;
    Slang::RefPtr<FileOutputStream> m_fileStream;
    Slang::String m_recordFileDirectory = Slang::Path::getCurrentPath();
    ParameterRecorder m_recorder;
};
} // namespace SlangRecord
#endif // RECORD_MANAGER_H
