// slang-artifact-util.cpp
#include "slang-artifact-util.h"

#include "../core/slang-castable.h"
#include "../core/slang-io.h"
#include "slang-artifact-desc-util.h"
#include "slang-artifact-impl.h"
#include "slang-artifact-representation-impl.h"

namespace Slang
{

static bool _checkSelf(ArtifactUtil::FindStyle findStyle)
{
    return Index(findStyle) <= Index(ArtifactUtil::FindStyle::SelfOrChildren);
}

static bool _checkChildren(ArtifactUtil::FindStyle findStyle)
{
    return Index(findStyle) >= Index(ArtifactUtil::FindStyle::SelfOrChildren);
}

static bool _checkRecursive(ArtifactUtil::FindStyle findStyle)
{
    return findStyle == ArtifactUtil::FindStyle::Recursive ||
           findStyle == ArtifactUtil::FindStyle::ChildrenRecursive;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ArtifactUtil !!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* static */ ComPtr<IArtifact> ArtifactUtil::createArtifact(
    const ArtifactDesc& desc,
    const char* name)
{
    auto artifact = createArtifact(desc);
    artifact->setName(name);
    return artifact;
}

/* static */ ComPtr<IArtifact> ArtifactUtil::createArtifact(const ArtifactDesc& desc)
{
    return Artifact::create(desc);
}

/* static */ ComPtr<IArtifact> ArtifactUtil::createArtifactForCompileTarget(
    SlangCompileTarget target)
{
    auto desc = ArtifactDescUtil::makeDescForCompileTarget(target);
    return createArtifact(desc);
}

/* static */ bool ArtifactUtil::isSignificant(IArtifact* artifact)
{
    return isSignificant(artifact->getDesc());
}

/* static */ bool ArtifactUtil::isSignificant(const ArtifactDesc& desc)
{
    // Containers are not significant as of themselves, they may contain something tho
    if (isDerivedFrom(desc.kind, ArtifactKind::Container))
    {
        return false;
    }

    // If it has no payload.. we are done
    if (desc.payload == ArtifactPayload::None || desc.payload == ArtifactPayload::Invalid)
    {
        return false;
    }

    // If it's binary like or assembly/source we it's significant
    if (isDerivedFrom(desc.kind, ArtifactKind::CompileBinary) ||
        desc.kind == ArtifactKind::Assembly || desc.kind == ArtifactKind::Source)
    {
        return true;
    }

    /* Hmm, we might want to have a base class for 'significant' payloads,
    where signifiance here means somewhat approximately 'the meat' of a compilation result,
    as contrasted with 'meta data', 'diagnostics etc'*/
    if (isDerivedFrom(desc.payload, ArtifactPayload::Metadata))
    {
        return false;
    }

    /* We currently can't write out diagnostics, so for now we'll say they are insignificant */
    if (isDerivedFrom(desc.payload, ArtifactPayload::Diagnostics))
    {
        return false;
    }

    return true;
}


/* static */ bool ArtifactUtil::isSignificant(IArtifact* artifact, void* data)
{
    SLANG_UNUSED(data);
    return isSignificant(artifact->getDesc());
}

/* static */ IArtifact* ArtifactUtil::findSignificant(IArtifact* artifact)
{
    return findArtifactByPredicate(
        artifact,
        FindStyle::SelfOrChildren,
        &ArtifactUtil::isSignificant,
        nullptr);
}

UnownedStringSlice ArtifactUtil::findPath(IArtifact* artifact)
{
    // If a name is set we'll just use that
    {
        const UnownedStringSlice name(artifact->getName());
        if (name.getLength())
        {
            return name;
        }
    }

    IPathArtifactRepresentation* bestRep = nullptr;

    // Look for a rep with a path. Prefer IExtFile because a IOSFile might be a temporary file
    for (auto rep : artifact->getRepresentations())
    {
        if (auto pathRep = as<IPathArtifactRepresentation>(rep))
        {
            if (pathRep->getPathType() == SLANG_PATH_TYPE_FILE &&
                (bestRep == nullptr || as<IExtFileArtifactRepresentation>(rep)))
            {
                bestRep = pathRep;
            }
        }
    }

    const UnownedStringSlice name =
        bestRep ? UnownedStringSlice(bestRep->getPath()) : UnownedStringSlice();
    return name.getLength() ? name : UnownedStringSlice();
}

/* static */ UnownedStringSlice ArtifactUtil::inferExtension(IArtifact* artifact)
{
    const UnownedStringSlice path = findPath(artifact);
    if (path.getLength())
    {
        auto ext = Path::getPathExt(UnownedStringSlice(path));
        if (ext.getLength())
        {
            return ext;
        }
    }
    return UnownedStringSlice();
}

/* static */ UnownedStringSlice ArtifactUtil::findName(IArtifact* artifact)
{
    const UnownedStringSlice path = findPath(artifact);
    const Index pos = Path::findLastSeparatorIndex(path);
    return (pos >= 0) ? path.tail(pos + 1) : path;
}

static SlangResult _calcInferred(
    IArtifact* artifact,
    const UnownedStringSlice& basePath,
    StringBuilder& outPath)
{
    auto ext = ArtifactUtil::inferExtension(artifact);

    // If no extension was determined by inferring, go with unknown
    if (ext.begin() == nullptr)
    {
        ext = toSlice("unknown");
    }

    outPath.clear();
    outPath.append(basePath);
    if (ext.getLength())
    {
        outPath.appendChar('.');
        outPath.append(ext);
    }
    return SLANG_OK;
}

/* static */ SlangResult ArtifactUtil::calcPath(
    IArtifact* artifact,
    const UnownedStringSlice& basePath,
    StringBuilder& outPath)
{
    if (ArtifactDescUtil::hasDefinedNameForDesc(artifact->getDesc()))
    {
        return ArtifactDescUtil::calcPathForDesc(artifact->getDesc(), basePath, outPath);
    }
    else
    {
        return _calcInferred(artifact, basePath, outPath);
    }
}

/* static */ SlangResult ArtifactUtil::calcName(
    IArtifact* artifact,
    const UnownedStringSlice& baseName,
    StringBuilder& outName)
{
    if (ArtifactDescUtil::hasDefinedNameForDesc(artifact->getDesc()))
    {
        return ArtifactDescUtil::calcNameForDesc(artifact->getDesc(), baseName, outName);
    }
    else
    {
        return _calcInferred(artifact, baseName, outName);
    }
}

static bool _isByDerivedDesc(IArtifact* artifact, void* data)
{
    const ArtifactDesc& desc = *(const ArtifactDesc*)data;
    return ArtifactDescUtil::isDescDerivedFrom(artifact->getDesc(), desc);
}

static bool _isDesc(IArtifact* artifact, void* data)
{
    const ArtifactDesc& desc = *(const ArtifactDesc*)data;
    return artifact->getDesc() == desc;
}

static bool _isName(IArtifact* artifact, void* data)
{
    const char* name = (const char*)data;
    const auto artifactName = artifact->getName();

    if (name == nullptr || artifactName == nullptr)
    {
        return name == artifactName;
    }
    return ::strcmp(name, artifactName) == 0;
}

/* static */ IArtifact* ArtifactUtil::findArtifactByDerivedDesc(
    IArtifact* artifact,
    FindStyle findStyle,
    const ArtifactDesc& desc)
{
    return findArtifactByPredicate(
        artifact,
        findStyle,
        _isByDerivedDesc,
        &const_cast<ArtifactDesc&>(desc));
}

/* static */ IArtifact* ArtifactUtil::findArtifactByName(
    IArtifact* artifact,
    FindStyle findStyle,
    const char* name)
{
    return findArtifactByPredicate(artifact, findStyle, _isName, const_cast<char*>(name));
}

/* static */ IArtifact* ArtifactUtil::findArtifactByDesc(
    IArtifact* artifact,
    FindStyle findStyle,
    const ArtifactDesc& desc)
{
    return findArtifactByPredicate(artifact, findStyle, _isDesc, &const_cast<ArtifactDesc&>(desc));
}

/* static */ IArtifact* ArtifactUtil::findArtifactByPredicate(
    IArtifact* artifact,
    FindStyle findStyle,
    FindFunc func,
    void* data)
{
    if (_checkSelf(findStyle) && func(artifact, data))
    {
        return artifact;
    }

    if (!_checkChildren(findStyle))
    {
        return nullptr;
    }

    // Expand the children so we can search them
    artifact->expandChildren();

    auto children = artifact->getChildren();
    if (children.count == 0)
    {
        return nullptr;
    }

    // Check the children
    for (auto child : children)
    {
        if (func(child, data))
        {
            return child;
        }
    }

    // If it's recursive, we check all the children of children
    if (_checkRecursive(findStyle))
    {
        for (auto child : children)
        {
            if (auto found =
                    findArtifactByPredicate(child, FindStyle::ChildrenRecursive, func, data))
            {
                return found;
            }
        }
    }

    return nullptr;
}

/* static */ void ArtifactUtil::addAssociated(
    IArtifact* artifact,
    IArtifactPostEmitMetadata* metadata)
{
    if (metadata)
    {
        auto metadataArtifact = ArtifactUtil::createArtifact(
            ArtifactDesc::make(ArtifactKind::Instance, ArtifactPayload::PostEmitMetadata));
        metadataArtifact->addRepresentation(metadata);
        artifact->addAssociated(metadataArtifact);
    }
}

/* static */ void ArtifactUtil::addAssociated(
    IArtifact* artifact,
    IArtifactDiagnostics* diagnostics)
{
    if (diagnostics)
    {
        auto diagnosticsArtifact = ArtifactUtil::createArtifact(
            ArtifactDesc::make(ArtifactKind::Instance, ArtifactPayload::Diagnostics));
        diagnosticsArtifact->addRepresentation(diagnostics);
        artifact->addAssociated(diagnosticsArtifact);
    }
}

} // namespace Slang
