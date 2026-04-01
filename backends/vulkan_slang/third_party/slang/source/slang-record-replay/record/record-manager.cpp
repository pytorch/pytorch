#include "record-manager.h"

#include "../../core/slang-io.h"
#include "../util/record-utility.h"

#include <sstream>
#include <thread>

namespace SlangRecord
{
RecordManager::RecordManager(uint64_t globalSessionHandle)
    : m_recorder(&m_memoryStream)
{
    std::stringstream ss;
    ss << "gs-" << globalSessionHandle << "-t-" << std::this_thread::get_id() << ".cap";

    m_recordFileDirectory = Slang::Path::combine(m_recordFileDirectory, "slang-record");
    if (!Slang::File::exists(m_recordFileDirectory))
    {
        if (!Slang::Path::createDirectoryRecursive(m_recordFileDirectory))
        {
            slangRecordLog(
                LogLevel::Error,
                "Fail to create directory: %s\n",
                m_recordFileDirectory.getBuffer());
        }
    }

    Slang::String recordFilePath =
        Slang::Path::combine(m_recordFileDirectory, Slang::String(ss.str().c_str()));
    m_fileStream = new FileOutputStream(recordFilePath);
}

void RecordManager::clearWithHeader(const ApiCallId& callId, uint64_t handleId)
{
    m_memoryStream.flush();
    FunctionHeader header;
    header.callId = callId;
    header.handleId = handleId;

    // write header to memory stream
    m_memoryStream.write(&header, sizeof(FunctionHeader));
}

void RecordManager::clearWithTailer()
{
    m_memoryStream.flush();
    FunctionTailer tailer;

    // write header to memory stream
    m_memoryStream.write(&tailer, sizeof(FunctionTailer));
}

ParameterRecorder* RecordManager::beginMethodRecord(const ApiCallId& callId, uint64_t handleId)
{
    clearWithHeader(callId, handleId);
    return &m_recorder;
}

ParameterRecorder* RecordManager::endMethodRecord()
{
    FunctionHeader* pHeader = const_cast<FunctionHeader*>(
        reinterpret_cast<const FunctionHeader*>(m_memoryStream.getData()));

    pHeader->dataSizeInBytes = m_memoryStream.getSizeInBytes() - sizeof(FunctionHeader);

    std::hash<std::thread::id> hasher;
    pHeader->threadId = hasher(std::this_thread::get_id());

    // write record data to file
    m_fileStream->write(m_memoryStream.getData(), m_memoryStream.getSizeInBytes());

    // take effect of the write
    m_fileStream->flush();

    // clear the memory stream
    m_memoryStream.flush();

    clearWithTailer();
    return &m_recorder;
}

void RecordManager::apendOutput()
{
    FunctionTailer* pTailer = const_cast<FunctionTailer*>(
        reinterpret_cast<const FunctionTailer*>(m_memoryStream.getData()));

    pTailer->dataSizeInBytes = (uint32_t)(m_memoryStream.getSizeInBytes() - sizeof(FunctionTailer));

    // write record data to file
    m_fileStream->write(m_memoryStream.getData(), m_memoryStream.getSizeInBytes());

    // take effect of the write
    m_fileStream->flush();

    // clear the memory stream
    m_memoryStream.flush();
}
} // namespace SlangRecord
