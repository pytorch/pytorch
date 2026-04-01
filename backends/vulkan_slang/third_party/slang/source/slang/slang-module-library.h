// slang-module-library.h
#ifndef SLANG_MODULE_LIBRARY_H
#define SLANG_MODULE_LIBRARY_H

#include "../compiler-core/slang-artifact-representation.h"
#include "slang-compiler.h"

namespace Slang
{

class IModuleLibrary : public IArtifactRepresentation
{
    SLANG_COM_INTERFACE(
        0x8f630911,
        0xea96,
        0x4075,
        {0xbf, 0x6b, 0xd2, 0xae, 0x96, 0xbe, 0xb6, 0xde});
};

// Class to hold information serialized in from a -r slang-lib/slang-module
class ModuleLibrary : public ComBaseObject, public IModuleLibrary
{
public:
    SLANG_COM_BASE_IUNKNOWN_ALL

    SLANG_CLASS_GUID(0x2f7412bd, 0x6154, 0x40a9, {0x89, 0xb3, 0x62, 0xe0, 0x24, 0x17, 0x24, 0xa1});

    // ICastable
    virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const Guid& guid) SLANG_OVERRIDE;

    // IArtifactRepresentation
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    createRepresentation(const Guid& typeGuid, ICastable** outCastable) SLANG_OVERRIDE
    {
        SLANG_UNUSED(typeGuid);
        SLANG_UNUSED(outCastable);
        return SLANG_E_NOT_IMPLEMENTED;
    }
    virtual SLANG_NO_THROW bool SLANG_MCALL exists() SLANG_OVERRIDE { return true; }

    List<FrontEndCompileRequest::ExtraEntryPointInfo> m_entryPoints;
    List<RefPtr<Module>> m_modules;

    void* getInterface(const Guid& uuid);
    void* getObject(const Guid& uuid);
};

SlangResult loadModuleLibrary(
    const Byte* inBytes,
    size_t bytesCount,
    String Path,
    EndToEndCompileRequest* req,
    ComPtr<IModuleLibrary>& outModule);

// Given a product make available as a module
SlangResult loadModuleLibrary(
    ArtifactKeep keep,
    IArtifact* artifact,
    String Path,
    EndToEndCompileRequest* req,
    ComPtr<IModuleLibrary>& outModule);

} // namespace Slang

#endif
