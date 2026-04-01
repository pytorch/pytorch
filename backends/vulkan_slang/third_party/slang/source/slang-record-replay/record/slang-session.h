#ifndef SLANG_SESSION_H
#define SLANG_SESSION_H

#include "../../core/slang-dictionary.h"
#include "../../core/slang-smart-pointer.h"
#include "../../slang/slang-compiler.h"
#include "record-manager.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang-module.h"
#include "slang.h"

namespace SlangRecord
{
using namespace Slang;
class SessionRecorder : public RefObject, public slang::ISession
{
public:
    SLANG_REF_OBJECT_IUNKNOWN_ALL
    ISlangUnknown* getInterface(const Guid& guid);

    explicit SessionRecorder(slang::ISession* session, RecordManager* recordManager);

    SLANG_NO_THROW slang::IGlobalSession* SLANG_MCALL getGlobalSession() override;
    SLANG_NO_THROW slang::IModule* SLANG_MCALL
    loadModule(const char* moduleName, slang::IBlob** outDiagnostics = nullptr) override;
    SLANG_NO_THROW slang::IModule* SLANG_MCALL loadModuleFromIRBlob(
        const char* moduleName,
        const char* path,
        slang::IBlob* source,
        slang::IBlob** outDiagnostics = nullptr) override;
    SLANG_NO_THROW slang::IModule* SLANG_MCALL loadModuleFromSource(
        const char* moduleName,
        const char* path,
        slang::IBlob* source,
        slang::IBlob** outDiagnostics = nullptr) override;
    SLANG_NO_THROW slang::IModule* SLANG_MCALL loadModuleFromSourceString(
        const char* moduleName,
        const char* path,
        const char* string,
        slang::IBlob** outDiagnostics = nullptr) override;
    SLANG_NO_THROW SlangResult SLANG_MCALL createCompositeComponentType(
        slang::IComponentType* const* componentTypes,
        SlangInt componentTypeCount,
        slang::IComponentType** outCompositeComponentType,
        ISlangBlob** outDiagnostics = nullptr) override;
    SLANG_NO_THROW slang::TypeReflection* SLANG_MCALL specializeType(
        slang::TypeReflection* type,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ISlangBlob** outDiagnostics = nullptr) override;
    SLANG_NO_THROW slang::TypeLayoutReflection* SLANG_MCALL getTypeLayout(
        slang::TypeReflection* type,
        SlangInt targetIndex = 0,
        slang::LayoutRules rules = slang::LayoutRules::Default,
        ISlangBlob** outDiagnostics = nullptr) override;
    SLANG_NO_THROW slang::TypeReflection* SLANG_MCALL getContainerType(
        slang::TypeReflection* elementType,
        slang::ContainerType containerType,
        ISlangBlob** outDiagnostics = nullptr) override;
    SLANG_NO_THROW slang::TypeReflection* SLANG_MCALL getDynamicType() override;
    SLANG_NO_THROW SlangResult SLANG_MCALL
    getTypeRTTIMangledName(slang::TypeReflection* type, ISlangBlob** outNameBlob) override;
    SLANG_NO_THROW SlangResult SLANG_MCALL getTypeConformanceWitnessMangledName(
        slang::TypeReflection* type,
        slang::TypeReflection* interfaceType,
        ISlangBlob** outNameBlob) override;
    SLANG_NO_THROW SlangResult SLANG_MCALL getTypeConformanceWitnessSequentialID(
        slang::TypeReflection* type,
        slang::TypeReflection* interfaceType,
        uint32_t* outId) override;
    SLANG_NO_THROW SlangResult SLANG_MCALL createTypeConformanceComponentType(
        slang::TypeReflection* type,
        slang::TypeReflection* interfaceType,
        slang::ITypeConformance** outConformance,
        SlangInt conformanceIdOverride,
        ISlangBlob** outDiagnostics) override;
    SLANG_NO_THROW SlangResult SLANG_MCALL
    createCompileRequest(SlangCompileRequest** outCompileRequest) override;
    SLANG_NO_THROW SlangInt SLANG_MCALL getLoadedModuleCount() override;
    SLANG_NO_THROW slang::IModule* SLANG_MCALL getLoadedModule(SlangInt index) override;
    SLANG_NO_THROW bool SLANG_MCALL
    isBinaryModuleUpToDate(const char* modulePath, slang::IBlob* binaryModuleBlob) override;

private:
    SLANG_FORCE_INLINE slang::ISession* asExternal(SessionRecorder* session)
    {
        return static_cast<slang::ISession*>(session);
    }

    // The IComponentType object is the record target, therefore `componentTypes` will not be
    // the actual component types, we have to use the COM interface to get the actual objects.
    SlangResult getActualComponentTypes(
        slang::IComponentType* const* componentTypes,
        SlangInt componentTypeCount,
        List<slang::IComponentType*>& outActualComponentTypes);

    IModuleRecorder* getModuleRecorder(slang::IModule* module);

    Slang::ComPtr<slang::ISession> m_actualSession;
    uint64_t m_sessionHandle = 0;

    Dictionary<slang::IModule*, IModuleRecorder*> m_mapModuleToRecord;
    List<ComPtr<IModuleRecorder>> m_moduleRecordersAlloation;
    RecordManager* m_recordManager = nullptr;
};
} // namespace SlangRecord

#endif // SLANG_SESSION_H
