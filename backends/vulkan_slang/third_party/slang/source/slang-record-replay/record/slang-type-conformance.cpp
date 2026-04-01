#include "slang-type-conformance.h"

#include "../util/record-utility.h"

namespace SlangRecord
{
TypeConformanceRecorder::TypeConformanceRecorder(
    SessionRecorder* sessionRecorder,
    slang::ITypeConformance* typeConformance,
    RecordManager* recordManager)
    : IComponentTypeRecorder(typeConformance, recordManager)
    , m_sessionRecorder(sessionRecorder)
    , m_actualTypeConformance(typeConformance)
{
    SLANG_RECORD_ASSERT(m_actualTypeConformance != nullptr);
    SLANG_RECORD_ASSERT(m_recordManager != nullptr);

    m_typeConformanceHandle = reinterpret_cast<uint64_t>(m_actualTypeConformance.get());
    slangRecordLog(LogLevel::Verbose, "%s: %p\n", __PRETTY_FUNCTION__, typeConformance);
}

ISlangUnknown* TypeConformanceRecorder::getInterface(const Guid& guid)
{
    if (guid == ITypeConformanceRecorder::getTypeGuid())
    {
        return static_cast<ITypeConformanceRecorder*>(this);
    }
    else
    {
        return nullptr;
    }
}
} // namespace SlangRecord
