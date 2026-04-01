#include "slang-composite-component-type.h"

#include "../util/record-utility.h"

namespace SlangRecord
{
CompositeComponentTypeRecorder::CompositeComponentTypeRecorder(
    SessionRecorder* sessionRecorder,
    slang::IComponentType* componentType,
    RecordManager* recordManager)
    : IComponentTypeRecorder(componentType, recordManager), m_sessionRecorder(sessionRecorder)
{
    slangRecordLog(LogLevel::Verbose, "%s: %p\n", __PRETTY_FUNCTION__, componentType);
}

ISlangUnknown* CompositeComponentTypeRecorder::getInterface(const Guid& guid)
{
    if (guid == CompositeComponentTypeRecorder::getTypeGuid())
    {
        return static_cast<ISlangUnknown*>(this);
    }
    return nullptr;
}
} // namespace SlangRecord
