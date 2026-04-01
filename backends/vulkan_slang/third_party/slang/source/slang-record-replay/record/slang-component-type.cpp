#include "slang-component-type.h"

#include "../util/record-utility.h"
#include "slang-composite-component-type.h"
#include "slang-session.h"

namespace SlangRecord
{
IComponentTypeRecorder::IComponentTypeRecorder(
    slang::IComponentType* componentType,
    RecordManager* recordManager)
    : m_actualComponentType(componentType), m_recordManager(recordManager)
{
    SLANG_RECORD_ASSERT(m_actualComponentType != nullptr);
    SLANG_RECORD_ASSERT(m_recordManager != nullptr);

    m_componentHandle = reinterpret_cast<uint64_t>(m_actualComponentType.get());
    slangRecordLog(LogLevel::Verbose, "%s: %p\n", __PRETTY_FUNCTION__, componentType);
}

SLANG_NO_THROW slang::ISession* IComponentTypeRecorder::getSession()
{
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);

    ApiCallId callId =
        static_cast<ApiCallId>(makeApiCallId(getClassId(), IComponentTypeMethodId::getSession));
    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(callId, m_componentHandle);
        recorder = m_recordManager->endMethodRecord();
    }

    slang::ISession* res = m_actualComponentType->getSession();

    {
        recorder->recordAddress(res);
        m_recordManager->apendOutput();
    }

    // instead of returning the actual session, we need to return the recorder
    SessionRecorder* sessionRecorder = getSessionRecorder();
    return static_cast<slang::ISession*>(sessionRecorder);
}

SLANG_NO_THROW slang::ProgramLayout* IComponentTypeRecorder::getLayout(
    SlangInt targetIndex,
    slang::IBlob** outDiagnostics)
{
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);

    ApiCallId callId =
        static_cast<ApiCallId>(makeApiCallId(getClassId(), IComponentTypeMethodId::getLayout));
    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(callId, m_componentHandle);
        recorder->recordInt64(targetIndex);
        recorder = m_recordManager->endMethodRecord();
    }

    slang::ProgramLayout* programLayout =
        m_actualComponentType->getLayout(targetIndex, outDiagnostics);

    {
        recorder->recordAddress(outDiagnostics ? *outDiagnostics : nullptr);
        recorder->recordAddress(programLayout);
        m_recordManager->apendOutput();
    }

    return programLayout;
}

SLANG_NO_THROW SlangInt IComponentTypeRecorder::getSpecializationParamCount()
{
    // No need to record this call as it is just a query.
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);
    SlangInt res = m_actualComponentType->getSpecializationParamCount();
    return res;
}

SLANG_NO_THROW SlangResult IComponentTypeRecorder::getEntryPointCode(
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    slang::IBlob** outCode,
    slang::IBlob** outDiagnostics)
{
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);

    ApiCallId callId = static_cast<ApiCallId>(
        makeApiCallId(getClassId(), IComponentTypeMethodId::getEntryPointCode));
    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(callId, m_componentHandle);
        recorder->recordInt64(entryPointIndex);
        recorder->recordInt64(targetIndex);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualComponentType->getEntryPointCode(
        entryPointIndex,
        targetIndex,
        outCode,
        outDiagnostics);

    {
        recorder->recordAddress(*outCode);
        recorder->recordAddress(outDiagnostics ? *outDiagnostics : nullptr);
        m_recordManager->apendOutput();
    }

    return res;
}

SLANG_NO_THROW SlangResult IComponentTypeRecorder::getTargetCode(
    SlangInt targetIndex,
    slang::IBlob** outCode,
    slang::IBlob** outDiagnostics)
{
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);

    ApiCallId callId =
        static_cast<ApiCallId>(makeApiCallId(getClassId(), IComponentTypeMethodId::getTargetCode));
    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(callId, m_componentHandle);
        recorder->recordInt64(targetIndex);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualComponentType->getTargetCode(targetIndex, outCode, outDiagnostics);

    {
        recorder->recordAddress(*outCode);
        recorder->recordAddress(outDiagnostics ? *outDiagnostics : nullptr);
        m_recordManager->apendOutput();
    }

    return res;
}

SLANG_NO_THROW SlangResult SLANG_MCALL IComponentTypeRecorder::getEntryPointMetadata(
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    slang::IMetadata** outMetadata,
    slang::IBlob** outDiagnostics)
{
    // No need to record this call.
    return m_actualComponentType
        ->getEntryPointMetadata(entryPointIndex, targetIndex, outMetadata, outDiagnostics);
}

SLANG_NO_THROW SlangResult SLANG_MCALL IComponentTypeRecorder::getTargetMetadata(
    SlangInt targetIndex,
    slang::IMetadata** outMetadata,
    slang::IBlob** outDiagnostics)
{
    // No need to record this call.
    return m_actualComponentType->getTargetMetadata(targetIndex, outMetadata, outDiagnostics);
}

SLANG_NO_THROW SlangResult IComponentTypeRecorder::getResultAsFileSystem(
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ISlangMutableFileSystem** outFileSystem)
{
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);

    ApiCallId callId = static_cast<ApiCallId>(
        makeApiCallId(getClassId(), IComponentTypeMethodId::getResultAsFileSystem));
    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(callId, m_componentHandle);
        recorder->recordInt64(entryPointIndex);
        recorder->recordInt64(targetIndex);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangResult res =
        m_actualComponentType->getResultAsFileSystem(entryPointIndex, targetIndex, outFileSystem);

    {
        recorder->recordAddress(*outFileSystem);
    }

    // TODO: We might need to wrap the file system object.
    return res;
}

SLANG_NO_THROW void IComponentTypeRecorder::getEntryPointHash(
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    slang::IBlob** outHash)
{
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);

    ApiCallId callId = static_cast<ApiCallId>(
        makeApiCallId(getClassId(), IComponentTypeMethodId::getEntryPointHash));
    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(callId, m_componentHandle);
        recorder->recordInt64(entryPointIndex);
        recorder->recordInt64(targetIndex);
        recorder = m_recordManager->endMethodRecord();
    }

    m_actualComponentType->getEntryPointHash(entryPointIndex, targetIndex, outHash);

    {
        recorder->recordAddress(*outHash);
        m_recordManager->apendOutput();
    }
}

SLANG_NO_THROW SlangResult IComponentTypeRecorder::specialize(
    slang::SpecializationArg const* specializationArgs,
    SlangInt specializationArgCount,
    slang::IComponentType** outSpecializedComponentType,
    ISlangBlob** outDiagnostics)
{
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);

    ApiCallId callId =
        static_cast<ApiCallId>(makeApiCallId(getClassId(), IComponentTypeMethodId::specialize));
    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(callId, m_componentHandle);
        recorder->recordInt64(specializationArgCount);
        recorder->recordStructArray(specializationArgs, specializationArgCount);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualComponentType->specialize(
        specializationArgs,
        specializationArgCount,
        outSpecializedComponentType,
        outDiagnostics);

    {
        recorder->recordAddress(*outSpecializedComponentType);
        recorder->recordAddress(outDiagnostics ? *outDiagnostics : nullptr);
        m_recordManager->apendOutput();
    }

    if (SLANG_SUCCEEDED(res))
    {
        // replaced output with our recorder
        *outSpecializedComponentType = getComponentTypeRecorder(*outSpecializedComponentType);
    }
    return res;
}

SLANG_NO_THROW SlangResult IComponentTypeRecorder::link(
    slang::IComponentType** outLinkedComponentType,
    ISlangBlob** outDiagnostics)
{
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);

    ApiCallId callId =
        static_cast<ApiCallId>(makeApiCallId(getClassId(), IComponentTypeMethodId::link));
    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(callId, m_componentHandle);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualComponentType->link(outLinkedComponentType, outDiagnostics);

    {
        recorder->recordAddress(*outLinkedComponentType);
        recorder->recordAddress(outDiagnostics ? *outDiagnostics : nullptr);
        m_recordManager->apendOutput();
    }

    if (SLANG_SUCCEEDED(res))
    {
        // replaced output with our recorder
        *outLinkedComponentType = getComponentTypeRecorder(*outLinkedComponentType);
    }
    return res;
}

SLANG_NO_THROW SlangResult IComponentTypeRecorder::getEntryPointHostCallable(
    int entryPointIndex,
    int targetIndex,
    ISlangSharedLibrary** outSharedLibrary,
    slang::IBlob** outDiagnostics)
{
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);

    ApiCallId callId = static_cast<ApiCallId>(
        makeApiCallId(getClassId(), IComponentTypeMethodId::getEntryPointHostCallable));
    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(callId, m_componentHandle);
        recorder->recordInt32(entryPointIndex);
        recorder->recordInt32(targetIndex);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualComponentType->getEntryPointHostCallable(
        entryPointIndex,
        targetIndex,
        outSharedLibrary,
        outDiagnostics);

    {
        recorder->recordAddress(*outSharedLibrary);
        recorder->recordAddress(outDiagnostics ? *outDiagnostics : nullptr);
        m_recordManager->apendOutput();
    }

    return res;
}

SLANG_NO_THROW SlangResult
IComponentTypeRecorder::renameEntryPoint(const char* newName, IComponentType** outEntryPoint)
{
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);

    ApiCallId callId = static_cast<ApiCallId>(
        makeApiCallId(getClassId(), IComponentTypeMethodId::renameEntryPoint));
    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(callId, m_componentHandle);
        recorder->recordString(newName);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualComponentType->renameEntryPoint(newName, outEntryPoint);

    {
        recorder->recordAddress(*outEntryPoint);
        m_recordManager->apendOutput();
    }

    // replaced output with our recorder
    // 'outEntryPoint' is not actually a IEntryPoint type, but a ComponentType type, so we
    // keep using CompositeComponentTypeRecorder to record it.
    if (SLANG_SUCCEEDED(res))
    {
        // replaced output with our recorder
        *outEntryPoint = getComponentTypeRecorder(*outEntryPoint);
    }
    return res;
}

SLANG_NO_THROW SlangResult IComponentTypeRecorder::linkWithOptions(
    IComponentType** outLinkedComponentType,
    uint32_t compilerOptionEntryCount,
    slang::CompilerOptionEntry* compilerOptionEntries,
    ISlangBlob** outDiagnostics)
{
    slangRecordLog(LogLevel::Verbose, "%s\n", __PRETTY_FUNCTION__);

    ApiCallId callId = static_cast<ApiCallId>(
        makeApiCallId(getClassId(), IComponentTypeMethodId::linkWithOptions));
    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(callId, m_componentHandle);
        recorder->recordUint32(compilerOptionEntryCount);
        recorder->recordStructArray(compilerOptionEntries, compilerOptionEntryCount);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualComponentType->linkWithOptions(
        outLinkedComponentType,
        compilerOptionEntryCount,
        compilerOptionEntries,
        outDiagnostics);

    {
        recorder->recordAddress(*outLinkedComponentType);
        recorder->recordAddress(outDiagnostics ? *outDiagnostics : nullptr);
        m_recordManager->apendOutput();
    }

    if (SLANG_SUCCEEDED(res))
    {
        // replaced output with our recorder
        *outLinkedComponentType = getComponentTypeRecorder(*outLinkedComponentType);
    }
    return res;
}

IComponentTypeRecorder* IComponentTypeRecorder::getComponentTypeRecorder(
    slang::IComponentType* componentTypes)
{
    IComponentTypeRecorder* recorder = nullptr;

    if (componentTypes)
    {
        if (m_mapComponentTypeToRecorder.tryGetValue(componentTypes, recorder))
        {
            ComPtr<IComponentTypeRecorder> result(recorder);
            return result.detach();
        }

        recorder = new CompositeComponentTypeRecorder(
            getSessionRecorder(),
            componentTypes,
            m_recordManager);
        ComPtr<IComponentTypeRecorder> result(recorder);
        m_componentTypeRecorderAlloation.add(result);
        m_mapComponentTypeToRecorder.add(componentTypes, result.detach());
    }
    return recorder;
}
} // namespace SlangRecord
