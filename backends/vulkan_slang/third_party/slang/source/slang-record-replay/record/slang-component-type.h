#ifndef SLANG_COMPONENT_TYPE_H
#define SLANG_COMPONENT_TYPE_H

#include "../../core/slang-smart-pointer.h"
#include "../../slang/slang-compiler.h"
#include "../util/record-utility.h"
#include "record-manager.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang.h"

namespace SlangRecord
{
using namespace Slang;
class SessionRecorder;

class IComponentTypeRecorder : public slang::IComponentType
{
public:
    explicit IComponentTypeRecorder(
        slang::IComponentType* componentType,
        RecordManager* recordManager);

    virtual SLANG_NO_THROW slang::ISession* SLANG_MCALL getSession() override;
    virtual SLANG_NO_THROW slang::ProgramLayout* SLANG_MCALL
    getLayout(SlangInt targetIndex = 0, slang::IBlob** outDiagnostics = nullptr) override;
    virtual SLANG_NO_THROW SlangInt SLANG_MCALL getSpecializationParamCount() override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointCode(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IBlob** outCode,
        slang::IBlob** outDiagnostics = nullptr) override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getResultAsFileSystem(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ISlangMutableFileSystem** outFileSystem) override;
    virtual SLANG_NO_THROW void SLANG_MCALL getEntryPointHash(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IBlob** outHash) override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL specialize(
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        slang::IComponentType** outSpecializedComponentType,
        ISlangBlob** outDiagnostics = nullptr) override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL link(
        slang::IComponentType** outLinkedComponentType,
        ISlangBlob** outDiagnostics = nullptr) override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointHostCallable(
        int entryPointIndex,
        int targetIndex,
        ISlangSharedLibrary** outSharedLibrary,
        slang::IBlob** outDiagnostics = 0) override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    renameEntryPoint(const char* newName, IComponentType** outEntryPoint) override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL linkWithOptions(
        IComponentType** outLinkedComponentType,
        uint32_t compilerOptionEntryCount,
        slang::CompilerOptionEntry* compilerOptionEntries,
        ISlangBlob** outDiagnostics = nullptr) override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getTargetCode(
        SlangInt targetIndex,
        slang::IBlob** outCode,
        slang::IBlob** outDiagnostics = nullptr) override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointMetadata(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IMetadata** outMetadata,
        slang::IBlob** outDiagnostics) override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getTargetMetadata(
        SlangInt targetIndex,
        slang::IMetadata** outMetadata,
        slang::IBlob** outDiagnostics = nullptr) override;

protected:
    virtual ApiClassId getClassId() = 0;
    virtual SessionRecorder* getSessionRecorder() = 0;
    Slang::ComPtr<slang::IComponentType> m_actualComponentType;
    uint64_t m_componentHandle = 0;
    RecordManager* m_recordManager = nullptr;

private:
    IComponentTypeRecorder* getComponentTypeRecorder(slang::IComponentType* componentTypes);

    Dictionary<slang::IComponentType*, IComponentTypeRecorder*> m_mapComponentTypeToRecorder;
    List<ComPtr<IComponentTypeRecorder>> m_componentTypeRecorderAlloation;
};
} // namespace SlangRecord

#endif // #ifndef SLANG_COMPONENT_TYPE_H
