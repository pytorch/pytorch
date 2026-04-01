// slang-artifact-impl.cpp
#include "slang-artifact-impl.h"

#include "../core/slang-castable.h"
#include "slang-artifact-desc-util.h"
#include "slang-artifact-handler-impl.h"
#include "slang-artifact-representation.h"
#include "slang-artifact-util.h"
#include "slang-slice-allocator.h"

namespace Slang
{

namespace
{ // anonymous

/* Get a view as a slice of *raw* pointers */
template<typename T>
SLANG_FORCE_INLINE ConstArrayView<T*> _getRawView(const List<ComPtr<T>>& in)
{
    return makeConstArrayView((T* const*)in.getBuffer(), in.getCount());
}

} // namespace

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Artifact !!!!!!!!!!!!!!!!!!!!!!!!!!! */

void* Artifact::castAs(const Guid& guid)
{
    if (auto ptr = getInterface(guid))
    {
        return ptr;
    }
    return getObject(guid);
}

void* Artifact::getInterface(const Guid& uuid)
{
    if (uuid == ISlangUnknown::getTypeGuid() || uuid == ICastable::getTypeGuid() ||
        uuid == IArtifact::getTypeGuid())
    {
        return static_cast<IArtifact*>(this);
    }
    return nullptr;
}

void* Artifact::getObject(const Guid& uuid)
{
    SLANG_UNUSED(uuid);
    return nullptr;
}

IArtifactHandler* Artifact::_getHandler()
{
    return m_handler ? m_handler : DefaultArtifactHandler::getSingleton();
}

void Artifact::_requireChildren()
{
    if (m_expandResult == SLANG_E_UNINITIALIZED)
    {
        const auto res = expandChildren();
        SLANG_UNUSED(res);

        SLANG_ASSERT(SLANG_SUCCEEDED(res) || res == SLANG_E_NOT_IMPLEMENTED);
    }
}

bool Artifact::exists()
{
    for (ICastable* rep : m_representations.getArrayView())
    {
        if (auto artifactRep = as<IArtifactRepresentation>(rep))
        {
            // It is an artifact rep and it exists, we are done
            if (artifactRep->exists())
            {
                return true;
            }
        }
        else
        {
            // If it's *not* IArtifactRepresentation derived, it's existance *is* a representation
            return true;
        }
    }

    return false;
}

SlangResult Artifact::requireFile(Keep keep, IOSFileArtifactRepresentation** outFileRep)
{
    auto handler = _getHandler();

    ComPtr<ICastable> castable;
    SLANG_RETURN_ON_FAIL(handler->getOrCreateRepresentation(
        this,
        IOSFileArtifactRepresentation::getTypeGuid(),
        keep,
        castable.writeRef()));

    auto fileRep = as<IOSFileArtifactRepresentation>(castable);
    fileRep->addRef();

    *outFileRep = fileRep;
    return SLANG_OK;
}

SlangResult Artifact::loadSharedLibrary(ArtifactKeep keep, ISlangSharedLibrary** outSharedLibrary)
{
    auto handler = _getHandler();

    ComPtr<ICastable> castable;
    SLANG_RETURN_ON_FAIL(handler->getOrCreateRepresentation(
        this,
        ISlangSharedLibrary::getTypeGuid(),
        keep,
        castable.writeRef()));

    auto lib = as<ISlangSharedLibrary>(castable);
    lib->addRef();

    *outSharedLibrary = lib;
    return SLANG_OK;
}

IArtifactHandler* Artifact::getHandler()
{
    return m_handler;
}

void Artifact::setHandler(IArtifactHandler* handler)
{
    m_handler = handler;
}

void Artifact::clear(IArtifact::ContainedKind kind)
{
    switch (kind)
    {
    case ContainedKind::Associated:
        m_associated.clear();
        break;
    case ContainedKind::Representation:
        m_representations.clear();
        break;
    case ContainedKind::Children:
        m_children.clear();
        break;
    default:
        break;
    }
}

void Artifact::removeAt(ContainedKind kind, Index i)
{
    switch (kind)
    {
    case ContainedKind::Associated:
        m_associated.removeAt(i);
        break;
    case ContainedKind::Representation:
        m_representations.removeAt(i);
        break;
    case ContainedKind::Children:
        m_children.removeAt(i);
        break;
    default:
        break;
    }
}

SlangResult Artifact::getOrCreateRepresentation(
    const Guid& typeGuid,
    ArtifactKeep keep,
    ICastable** outCastable)
{
    auto handler = _getHandler();
    return handler->getOrCreateRepresentation(this, typeGuid, keep, outCastable);
}

SlangResult Artifact::loadBlob(Keep keep, ISlangBlob** outBlob)
{
    auto handler = _getHandler();

    ComPtr<ICastable> castable;
    SLANG_RETURN_ON_FAIL(handler->getOrCreateRepresentation(
        this,
        ISlangBlob::getTypeGuid(),
        keep,
        castable.writeRef()));

    ISlangBlob* blob = as<ISlangBlob>(castable);
    blob->addRef();

    *outBlob = blob;
    return SLANG_OK;
}

void Artifact::addAssociated(IArtifact* artifact)
{
    SLANG_ASSERT(artifact);
    m_associated.add(ComPtr<IArtifact>(artifact));
}

static void* _findRepresentation(const ConstArrayView<ICastable*>& castables, const Guid& guid)
{
    for (const auto& cur : castables)
    {
        if (auto ptr = cur->castAs(guid))
        {
            return ptr;
        }
    }
    return nullptr;
}

static void* _findRepresentation(const ConstArrayView<IArtifact*>& artifacts, const Guid& guid)
{
    for (auto child : artifacts)
    {
        if (auto rep = child->findRepresentation(Artifact::ContainedKind::Representation, guid))
        {
            return rep;
        }
    }
    return nullptr;
}

void* Artifact::findRepresentation(ContainedKind kind, const Guid& guid)
{
    switch (kind)
    {
    case ContainedKind::Associated:
        return _findRepresentation(_getRawView(m_associated), guid);
    case ContainedKind::Representation:
        return _findRepresentation(_getRawView(m_representations), guid);
    case ContainedKind::Children:
        {
            _requireChildren();
            return _findRepresentation(_getRawView(m_children), guid);
        }
    }
    return nullptr;
}

Slice<IArtifact*> Artifact::getAssociated()
{
    return SliceUtil::asSlice(m_associated);
}

void Artifact::addRepresentation(ICastable* castable)
{
    SLANG_ASSERT(castable);

    auto view = _getRawView(m_representations);
    if (view.indexOf(castable) >= 0)
    {
        SLANG_ASSERT_FAILURE("Already have this representation");
        return;
    }

    m_representations.add(ComPtr<ICastable>(castable));
}

void Artifact::addRepresentationUnknown(ISlangUnknown* unk)
{
    SLANG_ASSERT(unk);

    {
        const auto view = makeConstArrayView(
            (ISlangUnknown* const*)m_representations.getBuffer(),
            m_representations.getCount());
        if (view.indexOf(unk) >= 0)
        {
            SLANG_ASSERT_FAILURE("Already have this representation");
            return;
        }
    }

    ComPtr<ICastable> castable;
    if (SLANG_SUCCEEDED(unk->queryInterface(SLANG_IID_PPV_ARGS(castable.writeRef()))) && castable)
    {
        if (_getRawView(m_representations).indexOf(castable) >= 0)
        {
            SLANG_ASSERT_FAILURE("Already have this representation");
            return;
        }
        m_representations.add(castable);
    }
    else
    {
        UnknownCastableAdapter* adapter = new UnknownCastableAdapter(unk);
        m_representations.add(ComPtr<ICastable>(adapter));
    }
}

Slice<ICastable*> Artifact::getRepresentations()
{
    return SliceUtil::asSlice(m_representations);
}

void Artifact::setChildren(IArtifact* const* children, Count count)
{
    m_expandResult = SLANG_OK;

    m_children.clearAndDeallocate();
    m_children.setCount(count);

    ComPtr<IArtifact>* dst = m_children.getBuffer();
    for (Index i = 0; i < count; ++i)
    {
        dst[i] = children[i];
    }
}

SlangResult Artifact::expandChildren()
{
    auto handler = _getHandler();
    return handler->expandChildren(this);
}

Slice<IArtifact*> Artifact::getChildren()
{
    _requireChildren();
    return SliceUtil::asSlice(m_children);
}

void Artifact::addChild(IArtifact* artifact)
{
    SLANG_ASSERT(artifact);
    SLANG_ASSERT(_getRawView(m_children).indexOf(artifact) < 0);

    _requireChildren();

    m_children.add(ComPtr<IArtifact>(artifact));
}

} // namespace Slang
