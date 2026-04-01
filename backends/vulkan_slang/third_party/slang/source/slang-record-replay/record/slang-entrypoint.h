#ifndef SLANG_ENTRY_POINT_H
#define SLANG_ENTRY_POINT_H

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

class IEntryPointRecorder : public slang::IEntryPoint, public RefObject
{
public:
    // SLANG_COM_INTERFACE(0xf4c1e23d, 0xb321, 0x4931, { 0x8f, 0x37, 0xf1, 0x22, 0x6a, 0xf9, 0x20,
    // 0x85 }) SLANG_REF_OBJECT_IUNKNOWN_ALL ISlangUnknown* getInterface(const Guid& guid) {
    // (void)guid; return nullptr;}
    SLANG_COM_INTERFACE(
        0xf4c1e23d,
        0xb321,
        0x4931,
        {0x8f, 0x37, 0xf1, 0x22, 0x6a, 0xf9, 0x20, 0x85})
};

class EntryPointRecorder : public IEntryPointRecorder, public IComponentTypeRecorder
{
    typedef IComponentTypeRecorder Super;

public:
    SLANG_REF_OBJECT_IUNKNOWN_ALL
    ISlangUnknown* getInterface(const Guid& guid);

    explicit EntryPointRecorder(
        SessionRecorder* sessionRecorder,
        slang::IEntryPoint* entryPoint,
        RecordManager* recordManager);

    // Interfaces for `IComponentType`
    virtual SLANG_NO_THROW slang::ISession* SLANG_MCALL getSession() override
    {
        return Super::getSession();
    }

    virtual SLANG_NO_THROW slang::ProgramLayout* SLANG_MCALL
    getLayout(SlangInt targetIndex = 0, slang::IBlob** outDiagnostics = nullptr) override
    {
        return Super::getLayout(targetIndex, outDiagnostics);
    }

    virtual SLANG_NO_THROW SlangInt SLANG_MCALL getSpecializationParamCount() override
    {
        return Super::getSpecializationParamCount();
    }

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointCode(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IBlob** outCode,
        slang::IBlob** outDiagnostics = nullptr) override
    {
        return Super::getEntryPointCode(entryPointIndex, targetIndex, outCode, outDiagnostics);
    }

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getTargetCode(
        SlangInt targetIndex,
        slang::IBlob** outCode,
        slang::IBlob** outDiagnostics = nullptr) override
    {
        return Super::getTargetCode(targetIndex, outCode, outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointMetadata(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IMetadata** outMetadata,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getEntryPointMetadata(
            entryPointIndex,
            targetIndex,
            outMetadata,
            outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getTargetMetadata(
        SlangInt targetIndex,
        slang::IMetadata** outMetadata,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getTargetMetadata(targetIndex, outMetadata, outDiagnostics);
    }

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getResultAsFileSystem(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ISlangMutableFileSystem** outFileSystem) override
    {
        return Super::getResultAsFileSystem(entryPointIndex, targetIndex, outFileSystem);
    }

    virtual SLANG_NO_THROW void SLANG_MCALL getEntryPointHash(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IBlob** outHash) override
    {
        return Super::getEntryPointHash(entryPointIndex, targetIndex, outHash);
    }

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL specialize(
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        slang::IComponentType** outSpecializedComponentType,
        ISlangBlob** outDiagnostics = nullptr) override
    {
        return Super::specialize(
            specializationArgs,
            specializationArgCount,
            outSpecializedComponentType,
            outDiagnostics);
    }

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL link(
        slang::IComponentType** outLinkedComponentType,
        ISlangBlob** outDiagnostics = nullptr) override
    {
        return Super::link(outLinkedComponentType, outDiagnostics);
    }

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointHostCallable(
        int entryPointIndex,
        int targetIndex,
        ISlangSharedLibrary** outSharedLibrary,
        slang::IBlob** outDiagnostics = 0) override
    {
        return Super::getEntryPointHostCallable(
            entryPointIndex,
            targetIndex,
            outSharedLibrary,
            outDiagnostics);
    }

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    renameEntryPoint(const char* newName, IComponentType** outEntryPoint) override
    {
        return Super::renameEntryPoint(newName, outEntryPoint);
    }

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL linkWithOptions(
        IComponentType** outLinkedComponentType,
        uint32_t compilerOptionEntryCount,
        slang::CompilerOptionEntry* compilerOptionEntries,
        ISlangBlob** outDiagnostics = nullptr) override
    {
        return Super::linkWithOptions(
            outLinkedComponentType,
            compilerOptionEntryCount,
            compilerOptionEntries,
            outDiagnostics);
    }

    // Interfaces for `IEntryPoint`
    virtual SLANG_NO_THROW slang::FunctionReflection* SLANG_MCALL getFunctionReflection() override;

    slang::IEntryPoint* getActualEntryPoint() const { return m_actualEntryPoint; }

protected:
    virtual ApiClassId getClassId() override { return ApiClassId::Class_IEntryPoint; }

    virtual SessionRecorder* getSessionRecorder() override { return m_sessionRecorder; }

private:
    SessionRecorder* m_sessionRecorder;
    Slang::ComPtr<slang::IEntryPoint> m_actualEntryPoint;
};
} // namespace SlangRecord
#endif // SLANG_ENTRY_POINT_H
