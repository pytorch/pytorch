#ifndef SLANG_MODULE_H
#define SLANG_MODULE_H

#include "../../core/slang-smart-pointer.h"
#include "../../slang/slang-compiler.h"
#include "record-manager.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang-entrypoint.h"
#include "slang.h"

namespace SlangRecord
{
using namespace Slang;
class SessionRecorder;

class IModuleRecorder : public slang::IModule, public RefObject
{
public:
    SLANG_COM_INTERFACE(
        0xb1802991,
        0x185a,
        0x4a03,
        {0xa7, 0x7e, 0x0c, 0x86, 0xe0, 0x68, 0x2a, 0xab})
};

class ModuleRecorder : public IModuleRecorder, public IComponentTypeRecorder
{
    typedef IComponentTypeRecorder Super;

public:
    SLANG_REF_OBJECT_IUNKNOWN_ALL
    ISlangUnknown* getInterface(const Guid& guid);

    explicit ModuleRecorder(
        SessionRecorder* sessionRecorder,
        slang::IModule* module,
        RecordManager* recordManager);

    // Interfaces for `IModule`
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    findEntryPointByName(char const* name, slang::IEntryPoint** outEntryPoint) override;
    virtual SLANG_NO_THROW SlangInt32 SLANG_MCALL getDefinedEntryPointCount() override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getDefinedEntryPoint(SlangInt32 index, slang::IEntryPoint** outEntryPoint) override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    serialize(ISlangBlob** outSerializedBlob) override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL writeToFile(char const* fileName) override;
    virtual SLANG_NO_THROW const char* SLANG_MCALL getName() override;
    virtual SLANG_NO_THROW const char* SLANG_MCALL getFilePath() override;
    virtual SLANG_NO_THROW const char* SLANG_MCALL getUniqueIdentity() override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL findAndCheckEntryPoint(
        char const* name,
        SlangStage stage,
        slang::IEntryPoint** outEntryPoint,
        ISlangBlob** outDiagnostics) override;
    virtual SLANG_NO_THROW SlangInt32 SLANG_MCALL getDependencyFileCount() override;
    virtual SLANG_NO_THROW char const* SLANG_MCALL getDependencyFilePath(SlangInt32 index) override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    disassemble(slang::IBlob** outDisassembly) override;

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

    virtual SLANG_NO_THROW slang::DeclReflection* SLANG_MCALL getModuleReflection() override;

    slang::IModule* getActualModule() const { return m_actualModule; }

protected:
    // `IComponentTypeRecorder` interface
    virtual ApiClassId getClassId() override { return ApiClassId::Class_IModule; }

    virtual SessionRecorder* getSessionRecorder() override { return m_sessionRecorder; }

private:
    IEntryPointRecorder* getEntryPointRecorder(slang::IEntryPoint* entryPoint);

    SessionRecorder* m_sessionRecorder;
    Slang::ComPtr<slang::IModule> m_actualModule;
    uint64_t m_moduleHandle = 0;
    RecordManager* m_recordManager = nullptr;

    // `IEntryPoint` can only be created from 'IModule', so we need to record it in
    // this class, and create a map such that we don't create new `EntryPointRecorder`
    // for the same `IEntryPoint`.
    Dictionary<slang::IEntryPoint*, IEntryPointRecorder*> m_mapEntryPointToRecord;
    List<ComPtr<IEntryPointRecorder>> m_entryPointsRecordAllocation;
};
} // namespace SlangRecord

#endif // SLANG_MODULE_H
