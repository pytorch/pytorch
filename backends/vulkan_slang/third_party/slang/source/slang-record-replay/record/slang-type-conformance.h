#ifndef SLANG_TYPE_CONFORMANCE_H
#define SLANG_TYPE_CONFORMANCE_H

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

class ITypeConformanceRecorder : public slang::ITypeConformance, public RefObject
{
public:
    SLANG_COM_INTERFACE(
        0x0e67d05d,
        0xee0a,
        0x41e1,
        {0xb5, 0xa3, 0x23, 0xe3, 0xb0, 0xec, 0x33, 0xf1})
};

class TypeConformanceRecorder : public ITypeConformanceRecorder, public IComponentTypeRecorder
{
    typedef IComponentTypeRecorder Super;

public:
    SLANG_REF_OBJECT_IUNKNOWN_ALL
    ISlangUnknown* getInterface(const Guid& guid);

    explicit TypeConformanceRecorder(
        SessionRecorder* sessionRecorder,
        slang::ITypeConformance* typeConformance,
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

    slang::ITypeConformance* getActualTypeConformance() const { return m_actualTypeConformance; }

protected:
    virtual ApiClassId getClassId() override { return ApiClassId::Class_ITypeConformance; }

    virtual SessionRecorder* getSessionRecorder() override { return m_sessionRecorder; }

private:
    SessionRecorder* m_sessionRecorder = nullptr;
    Slang::ComPtr<slang::ITypeConformance> m_actualTypeConformance;
    uint64_t m_typeConformanceHandle = 0;
    RecordManager* m_recordManager = nullptr;
};
} // namespace SlangRecord
#endif // SLANG_TYPE_CONFORMANCE_H
