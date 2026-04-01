#include "slang-module.h"

#include "../util/record-utility.h"
#include "slang-session.h"

namespace SlangRecord
{
ModuleRecorder::ModuleRecorder(
    SessionRecorder* sessionRecorder,
    slang::IModule* module,
    RecordManager* recordManager)
    : IComponentTypeRecorder(module, recordManager)
    , m_sessionRecorder(sessionRecorder)
    , m_actualModule(module)
    , m_recordManager(recordManager)
{
    SLANG_RECORD_ASSERT(m_actualModule != nullptr);
    SLANG_RECORD_ASSERT(m_recordManager != nullptr);

    m_moduleHandle = reinterpret_cast<uint64_t>(m_actualModule.get());
    slangRecordLog(LogLevel::Verbose, "%s: %p\n", __PRETTY_FUNCTION__, module);
}

ISlangUnknown* ModuleRecorder::getInterface(const Guid& guid)
{
    if (guid == IModuleRecorder::getTypeGuid())
        return static_cast<IModuleRecorder*>(this);
    else
        return nullptr;
}

SLANG_NO_THROW slang::DeclReflection* ModuleRecorder::getModuleReflection()
{
    // No need to record this call as it is just a query.
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);
    slang::DeclReflection* res = (slang::DeclReflection*)m_actualModule->getModuleReflection();
    return res;
}

SLANG_NO_THROW SlangResult
ModuleRecorder::findEntryPointByName(char const* name, slang::IEntryPoint** outEntryPoint)
{
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IModule_findEntryPointByName,
            m_moduleHandle);
        recorder->recordString(name);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualModule->findEntryPointByName(name, outEntryPoint);

    {
        recorder->recordAddress(*outEntryPoint);
        m_recordManager->apendOutput();
    }

    if (SLANG_OK == res)
    {
        IEntryPointRecorder* entryPointRecord = getEntryPointRecorder(*outEntryPoint);
        *outEntryPoint = static_cast<slang::IEntryPoint*>(entryPointRecord);
    }
    return res;
}

SLANG_NO_THROW SlangInt32 ModuleRecorder::getDefinedEntryPointCount()
{
    // No need to record this call as it is just a query.
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);
    SlangInt32 res = m_actualModule->getDefinedEntryPointCount();
    return res;
}

SLANG_NO_THROW SlangResult
ModuleRecorder::getDefinedEntryPoint(SlangInt32 index, slang::IEntryPoint** outEntryPoint)
{
    // This call is to find the existing entry point, so it has been created already. Therefore, we
    // don't create a new one and assert the error if it is not found in our map.
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IModule_getDefinedEntryPoint,
            m_moduleHandle);
        recorder->recordInt32(index);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualModule->getDefinedEntryPoint(index, outEntryPoint);

    {
        recorder->recordAddress(*outEntryPoint);
        m_recordManager->apendOutput();
    }

    if (*outEntryPoint)
    {
        IEntryPointRecorder* entryPointRecord = nullptr;
        bool ret = m_mapEntryPointToRecord.tryGetValue(*outEntryPoint, entryPointRecord);
        if (!ret)
        {
            SLANG_RECORD_ASSERT(!"Entrypoint not found in mapEntryPointToRecord");
        }
        ComPtr<slang::IEntryPoint> result(static_cast<slang::IEntryPoint*>(entryPointRecord));
        *outEntryPoint = result.detach();
    }
    else
        *outEntryPoint = nullptr;

    return res;
}

SLANG_NO_THROW SlangResult ModuleRecorder::serialize(ISlangBlob** outSerializedBlob)
{
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(ApiCallId::IModule_serialize, m_moduleHandle);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualModule->serialize(outSerializedBlob);

    {
        recorder->recordAddress(*outSerializedBlob);
        m_recordManager->apendOutput();
    }

    return res;
}

SLANG_NO_THROW SlangResult ModuleRecorder::writeToFile(char const* fileName)
{
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder =
            m_recordManager->beginMethodRecord(ApiCallId::IModule_writeToFile, m_moduleHandle);
        recorder->recordString(fileName);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualModule->writeToFile(fileName);
    return res;
}

SLANG_NO_THROW const char* ModuleRecorder::getName()
{
    // No need to record this call as it is just a query.
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);
    const char* res = m_actualModule->getName();
    return res;
}

SLANG_NO_THROW const char* ModuleRecorder::getFilePath()
{
    // No need to record this call as it is just a query.
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);
    const char* res = m_actualModule->getFilePath();
    return res;
}

SLANG_NO_THROW const char* ModuleRecorder::getUniqueIdentity()
{
    // No need to record this call as it is just a query.
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);
    const char* res = m_actualModule->getUniqueIdentity();
    return res;
}

SLANG_NO_THROW SlangResult ModuleRecorder::findAndCheckEntryPoint(
    char const* name,
    SlangStage stage,
    slang::IEntryPoint** outEntryPoint,
    ISlangBlob** outDiagnostics)
{
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IModule_findAndCheckEntryPoint,
            m_moduleHandle);
        recorder->recordString(name);
        recorder->recordEnumValue(stage);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangResult res =
        m_actualModule->findAndCheckEntryPoint(name, stage, outEntryPoint, outDiagnostics);

    {
        recorder->recordAddress(*outEntryPoint);
        recorder->recordAddress(outDiagnostics ? *outDiagnostics : nullptr);
        m_recordManager->apendOutput();
    }

    if (SLANG_OK == res)
    {
        IEntryPointRecorder* entryPointRecord = getEntryPointRecorder(*outEntryPoint);
        *outEntryPoint = static_cast<slang::IEntryPoint*>(entryPointRecord);
    }
    return res;
}

SLANG_NO_THROW SlangInt32 ModuleRecorder::getDependencyFileCount()
{
    // No need to record this call as it is just a query.
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);
    SlangInt32 res = m_actualModule->getDependencyFileCount();
    return res;
}

SLANG_NO_THROW char const* ModuleRecorder::getDependencyFilePath(SlangInt32 index)
{
    // No need to record this call as it is just a query.
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);
    const char* res = m_actualModule->getDependencyFilePath(index);
    return res;
}

IEntryPointRecorder* ModuleRecorder::getEntryPointRecorder(slang::IEntryPoint* entryPoint)
{
    IEntryPointRecorder* entryPointRecord = nullptr;
    bool ret = m_mapEntryPointToRecord.tryGetValue(entryPoint, entryPointRecord);
    if (!ret)
    {
        entryPointRecord = new EntryPointRecorder(m_sessionRecorder, entryPoint, m_recordManager);
        Slang::ComPtr<IEntryPointRecorder> result(entryPointRecord);

        m_entryPointsRecordAllocation.add(result);
        m_mapEntryPointToRecord.add(entryPoint, result.detach());
        return entryPointRecord;
    }
    else
    {
        Slang::ComPtr<IEntryPointRecorder> result(entryPointRecord);
        return result.detach();
    }
}

SlangResult ModuleRecorder::disassemble(ISlangBlob** outBlob)
{
    // No need to record this call as it is just a query.
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);
    auto res = m_actualModule->disassemble(outBlob);
    return res;
}
} // namespace SlangRecord
