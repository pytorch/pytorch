#include "slang-entrypoint.h"

#include "../util/record-utility.h"

namespace SlangRecord
{
EntryPointRecorder::EntryPointRecorder(
    SessionRecorder* sessionRecorder,
    slang::IEntryPoint* entryPoint,
    RecordManager* recordManager)
    : IComponentTypeRecorder(entryPoint, recordManager)
    , m_sessionRecorder(sessionRecorder)
    , m_actualEntryPoint(entryPoint)
{
    SLANG_RECORD_ASSERT(m_actualEntryPoint != nullptr);
    slangRecordLog(LogLevel::Verbose, "%s: %p\n", __PRETTY_FUNCTION__, entryPoint);
}

ISlangUnknown* EntryPointRecorder::getInterface(const Guid& guid)
{
    if (guid == IEntryPointRecorder::getTypeGuid())
    {
        return static_cast<IEntryPointRecorder*>(this);
    }
    else
        return nullptr;
}

SLANG_NO_THROW slang::FunctionReflection* EntryPointRecorder::getFunctionReflection()
{
    return m_actualEntryPoint->getFunctionReflection();
}

} // namespace SlangRecord
