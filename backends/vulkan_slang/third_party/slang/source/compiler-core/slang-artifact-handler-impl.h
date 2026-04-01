// slang-artifact-handler-impl.h
#ifndef SLANG_ARTIFACT_HANDLER_IMPL_H
#define SLANG_ARTIFACT_HANDLER_IMPL_H

#include "../core/slang-com-object.h"
#include "slang-artifact-representation.h"
#include "slang-artifact.h"

namespace Slang
{

class DefaultArtifactHandler : public ComBaseObject, public IArtifactHandler
{
public:
    SLANG_NO_THROW uint32_t SLANG_MCALL addRef() SLANG_OVERRIDE { return 1; }
    SLANG_NO_THROW uint32_t SLANG_MCALL release() SLANG_OVERRIDE { return 1; }
    SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(SlangUUID const& uuid, void** outObject)
        SLANG_OVERRIDE;

    // ICastable
    SLANG_NO_THROW void* SLANG_MCALL castAs(const Guid& guid) SLANG_OVERRIDE;

    // IArtifactHandler
    SLANG_NO_THROW SlangResult SLANG_MCALL expandChildren(IArtifact* container) SLANG_OVERRIDE;
    SLANG_NO_THROW SlangResult SLANG_MCALL getOrCreateRepresentation(
        IArtifact* artifact,
        const Guid& guid,
        ArtifactKeep keep,
        ICastable** outCastable) SLANG_OVERRIDE;

    static IArtifactHandler* getSingleton() { return &g_singleton; }

protected:
    SlangResult _loadSharedLibrary(IArtifact* artifact, ISlangSharedLibrary** outSharedLibrary);
    SlangResult _createOSFile(
        IArtifact* artifact,
        ArtifactKeep intermediateKeep,
        IOSFileArtifactRepresentation** outFileRep);

    void* getInterface(const Guid& uuid);
    void* getObject(const Guid& uuid);

    SlangResult _addRepresentation(
        IArtifact* artifact,
        ArtifactKeep keep,
        ISlangUnknown* rep,
        ICastable** outCastable);
    SlangResult _addRepresentation(
        IArtifact* artifact,
        ArtifactKeep keep,
        ICastable* castable,
        ICastable** outCastable);

    static DefaultArtifactHandler g_singleton;
};

} // namespace Slang

#endif
