#ifndef SLANG_COMPOSITE_COMPONENT_TYPE_H
#define SLANG_COMPOSITE_COMPONENT_TYPE_H

#include "../../core/slang-dictionary.h"
#include "../../core/slang-smart-pointer.h"
#include "../../slang/slang-compiler.h"
#include "record-manager.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang-component-type.h"
#include "slang.h"

namespace SlangRecord
{
using namespace Slang;
class SessionRecorder;

class CompositeComponentTypeRecorder : public IComponentTypeRecorder, public RefObject
{
public:
    SLANG_COM_INTERFACE(
        0x354f30a0,
        0x3662,
        0x4147,
        {0xa2, 0x5d, 0x9b, 0xc6, 0x95, 0x73, 0x8e, 0x07})

    SLANG_REF_OBJECT_IUNKNOWN_ALL
    ISlangUnknown* getInterface(const Guid& guid);

    explicit CompositeComponentTypeRecorder(
        SessionRecorder* sessionRecorder,
        slang::IComponentType* componentType,
        RecordManager* recordManager);

    slang::IComponentType* getActualCompositeComponentType() const { return m_actualComponentType; }

protected:
    virtual ApiClassId getClassId() override { return ApiClassId::Class_ICompositeComponentType; }

    virtual SessionRecorder* getSessionRecorder() override { return m_sessionRecorder; }

private:
    SessionRecorder* m_sessionRecorder = nullptr;
};
} // namespace SlangRecord
#endif // SLANG_COMPOSITE_COMPONENT_TYPE_H
