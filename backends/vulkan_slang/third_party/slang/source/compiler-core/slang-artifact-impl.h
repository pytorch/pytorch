// slang-artifact-impl.h
#ifndef SLANG_ARTIFACT_IMPL_H
#define SLANG_ARTIFACT_IMPL_H

#include "../core/slang-com-object.h"
#include "slang-artifact.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"

namespace Slang
{

/*
Discussion:

Another issue occurs around wanting to hold multiple kernels within a container. The problem here is
that although through the desc we can identify what target a kernel is for, there is no way of
telling what stage it is for.

When discussing the idea of a shader cache, one idea was to use a ISlangFileSystem (which could
actually be a zip, or directory or in memory rep) as the main structure. Within this it can contain
kernels, and then a json manifest can describe what each of these actually are.

This all 'works', in that we can add an element of ISlangFileSystem with a desc of Container. Code
that uses this can then go through the process of finding, and getting the blob, and find from the
manifest what it means. That does sound a little tedious though. Perhaps we just have an interface
that handles this detail, such that we search for that first. That interface is just attached to the
artifact as an element.
*/
class Artifact : public ComBaseObject, public IArtifact
{
public:
    SLANG_COM_BASE_IUNKNOWN_ALL

    /// ICastable
    virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const Guid& guid) SLANG_OVERRIDE;

    /// IArtifact impl
    virtual SLANG_NO_THROW Desc SLANG_MCALL getDesc() SLANG_OVERRIDE { return m_desc; }
    virtual SLANG_NO_THROW bool SLANG_MCALL exists() SLANG_OVERRIDE;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL loadBlob(Keep keep, ISlangBlob** outBlob)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    requireFile(Keep keep, IOSFileArtifactRepresentation** outFileRep) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    loadSharedLibrary(ArtifactKeep keep, ISlangSharedLibrary** outSharedLibrary) SLANG_OVERRIDE;

    virtual SLANG_NO_THROW const char* SLANG_MCALL getName() SLANG_OVERRIDE
    {
        return m_name.getBuffer();
    }
    virtual SLANG_NO_THROW void SLANG_MCALL setName(const char* name) SLANG_OVERRIDE
    {
        m_name = name;
    }

    virtual SLANG_NO_THROW void SLANG_MCALL addAssociated(IArtifact* artifact) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW Slice<IArtifact*> SLANG_MCALL getAssociated() SLANG_OVERRIDE;

    virtual SLANG_NO_THROW void SLANG_MCALL addRepresentation(ICastable* castable) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL addRepresentationUnknown(ISlangUnknown* rep)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW Slice<ICastable*> SLANG_MCALL getRepresentations() SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getOrCreateRepresentation(
        const Guid& typeGuid,
        ArtifactKeep keep,
        ICastable** outCastable) SLANG_OVERRIDE;

    virtual SLANG_NO_THROW IArtifactHandler* SLANG_MCALL getHandler() SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setHandler(IArtifactHandler* handler) SLANG_OVERRIDE;

    virtual SLANG_NO_THROW Slice<IArtifact*> SLANG_MCALL getChildren() SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getExpandChildrenResult() SLANG_OVERRIDE
    {
        return m_expandResult;
    }
    virtual SLANG_NO_THROW void SLANG_MCALL setChildren(IArtifact* const* children, Count count)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL expandChildren() SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL addChild(IArtifact* artifact) SLANG_OVERRIDE;

    virtual SLANG_NO_THROW void* SLANG_MCALL findRepresentation(ContainedKind kind, const Guid& unk)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL clear(ContainedKind kind) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL removeAt(ContainedKind kind, Index i) SLANG_OVERRIDE;

    static ComPtr<IArtifact> create(const Desc& desc)
    {
        return ComPtr<IArtifact>(new Artifact(desc));
    }
    static ComPtr<IArtifact> create(const Desc& desc, const UnownedStringSlice& name)
    {
        return ComPtr<IArtifact>(new Artifact(desc, name));
    }

protected:
    /// Ctor
    Artifact(const Desc& desc, const UnownedStringSlice& name)
        : m_desc(desc), m_name(name)
    {
    }
    Artifact(const Desc& desc)
        : m_desc(desc)
    {
    }

    IArtifactHandler* _getHandler();
    void _requireChildren();

    void* getInterface(const Guid& uuid);
    void* getObject(const Guid& uuid);

    Desc m_desc;   ///< Description of the artifact
    String m_name; ///< Name of this artifact
    SlangResult m_expandResult = SLANG_E_UNINITIALIZED;

    ComPtr<IArtifactHandler>
        m_handler; ///< The handler. Can be nullptr and then default handler is used.

    List<ComPtr<ICastable>> m_representations; ///< All the representation of this artifact
    List<ComPtr<IArtifact>> m_associated;      ///< All the items associated with this artifact
    List<ComPtr<IArtifact>> m_children;        ///< All the child artifacts owned
};

} // namespace Slang

#endif
