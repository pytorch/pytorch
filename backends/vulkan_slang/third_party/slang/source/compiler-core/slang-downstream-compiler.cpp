// slang-downstream-compiler.cpp
#include "slang-downstream-compiler.h"

#include "../core/slang-blob.h"
#include "../core/slang-castable.h"
#include "../core/slang-char-util.h"
#include "../core/slang-common.h"
#include "../core/slang-io.h"
#include "../core/slang-shared-library.h"
#include "../core/slang-string-util.h"
#include "../core/slang-type-text-util.h"
#include "slang-artifact-associated-impl.h"
#include "slang-artifact-desc-util.h"
#include "slang-artifact-helper.h"
#include "slang-artifact-impl.h"
#include "slang-artifact-representation-impl.h"
#include "slang-artifact-util.h"
#include "slang-com-helper.h"

namespace Slang
{

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DownstreamCompilerBase !!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

SlangResult DownstreamCompilerBase::convert(
    IArtifact* from,
    const ArtifactDesc& to,
    IArtifact** outArtifact)
{
    SLANG_UNUSED(from);
    SLANG_UNUSED(to);
    SLANG_UNUSED(outArtifact);

    return SLANG_E_NOT_AVAILABLE;
}

void* DownstreamCompilerBase::castAs(const Guid& guid)
{
    if (auto ptr = getInterface(guid))
    {
        return ptr;
    }
    return getObject(guid);
}

void* DownstreamCompilerBase::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ICastable::getTypeGuid() ||
        guid == IDownstreamCompiler::getTypeGuid())
    {
        return static_cast<IDownstreamCompiler*>(this);
    }

    return nullptr;
}

void* DownstreamCompilerBase::getObject(const Guid& guid)
{
    SLANG_UNUSED(guid);
    return nullptr;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CommandLineDownstreamCompiler !!!!!!!!!!!!!!!!!!!!!!*/

SlangResult CommandLineDownstreamCompiler::compile(
    const CompileOptions& inOptions,
    IArtifact** outArtifact)
{
    if (!isVersionCompatible(inOptions))
    {
        // Not possible to compile with this version of the interface.
        return SLANG_E_NOT_IMPLEMENTED;
    }

    CompileOptions options = getCompatibleVersion(&inOptions);

    // Copy the command line options
    CommandLine cmdLine(m_cmdLine);

    // Work out the ArtifactDesc
    const auto targetDesc = ArtifactDescUtil::makeDescForCompileTarget(options.targetType);

    auto helper = DefaultArtifactHelper::getSingleton();

    List<ComPtr<IArtifact>> artifactList;

    // It may be necessary to produce a temporary file 'lock file'.
    ComPtr<IOSFileArtifactRepresentation> lockFile;

    // The allocator can be used for items that are not kept in scope by the options
    String modulePath;

    // If no module path is set we will need to generate one
    if (options.modulePath.count == 0)
    {
        // We could use the path to the source, or use the source name/paths as defined on the
        // artifact For now we just go with a lock file based on "slang-generated".
        SLANG_RETURN_ON_FAIL(
            helper->createLockFile(asCharSlice(toSlice("slang-generated")), lockFile.writeRef()));

        auto lockArtifact = Artifact::create(
            ArtifactDesc::make(ArtifactKind::Base, ArtifactPayload::Lock, ArtifactStyle::None));
        lockArtifact->addRepresentation(lockFile);

        artifactList.add(lockArtifact);

        // Add the source files such that they can exist
        modulePath = lockFile->getPath();

        options.modulePath = SliceUtil::asTerminatedCharSlice(modulePath);
    }

    // Append command line args to the end of cmdLine using the target specific function for the
    // specified options
    SLANG_RETURN_ON_FAIL(calcArgs(options, cmdLine));

    // The 'productArtifact' is the main product produced from the compilation - the
    // executable/sharedlibrary/object etc
    ComPtr<IArtifact> productArtifact;
    {
        List<ComPtr<IArtifact>> artifacts;
        SLANG_RETURN_ON_FAIL(
            calcCompileProducts(options, DownstreamProductFlag::All, lockFile, artifacts));

        for (IArtifact* artifact : artifacts)
        {
            // The main artifact must be in the list, so add it if we find it
            if (artifact->getDesc() == targetDesc)
            {
                SLANG_ASSERT(productArtifact == nullptr);
                productArtifact = artifact;
            }

            artifactList.add(ComPtr<IArtifact>(artifact));
        }
    }

    SLANG_ASSERT(productArtifact);
    // Somethings gone wrong if we don't find the main artifact
    if (!productArtifact)
    {
        return SLANG_FAIL;
    }

    ExecuteResult exeRes;

#if 0
    // Test
    {
        String line = ProcessUtil::getCommandLineString(cmdLine);
        printf("%s", line.getBuffer());
    }
#endif

    SLANG_RETURN_ON_FAIL(ProcessUtil::execute(cmdLine, exeRes));

#if 0
    {
        printf("stdout=\"%s\"\nstderr=\"%s\"\nret=%d\n", exeRes.standardOutput.getBuffer(), exeRes.standardError.getBuffer(), int(exeRes.resultCode));
    }
#endif

    // Go through the list of artifacts in the artifactList and check if they exist.
    //
    // This is useful because `calcCompileProducts` is conservative and may produce artifacts for
    // products that aren't actually produced, by the compilation.
    {

        Count count = artifactList.getCount();
        for (Index i = 0; i < count; ++i)
        {
            IArtifact* artifact = artifactList[i];

            if (!artifact->exists())
            {
                // We should find a file rep and if we do we can disown it. Disowning will mean
                // when scope is lost the rep won't try and delete the (apparently non existing)
                // backing file.
                if (auto fileRep = findRepresentation<IOSFileArtifactRepresentation>(artifact))
                {
                    fileRep->disown();
                }

                // If the main artifact doesn't exist, we don't have a main artifact
                if (artifact == productArtifact)
                {
                    productArtifact.setNull();
                }

                // Remove from the list
                artifactList.removeAt(i);
                --count;
                --i;
            }
        }
    }

    // Add all of the source artifacts, that are temporary on the file system, such that they can
    // stay in scope for debugging
    for (auto sourceArtifact : options.sourceArtifacts)
    {
        if (auto fileRep = findRepresentation<IOSFileArtifactRepresentation>(sourceArtifact))
        {
            // If it has a lock file we can assume it's a temporary
            if (fileRep->getLockFile())
            {
                artifactList.add(ComPtr<IArtifact>(sourceArtifact));
            }
        }
    }

    // Create the result artifact
    auto artifact = ArtifactUtil::createArtifact(targetDesc);

    // Createa the diagnostics
    auto diagnostics = ArtifactDiagnostics::create();

    SLANG_RETURN_ON_FAIL(parseOutput(exeRes, diagnostics));

    ArtifactUtil::addAssociated(artifact, diagnostics);

    // Find the rep from the 'main' artifact, we'll just use the same representation on the output
    // artifact. Sharing is desirable, because the rep owns the file.
    if (auto fileRep = productArtifact
                           ? findRepresentation<IOSFileArtifactRepresentation>(productArtifact)
                           : nullptr)
    {
        artifact->addRepresentation(fileRep);
    }

    // Add the artifact list if there is anything in it
    if (artifactList.getCount())
    {
        // Holds all of the artifacts that are relatated to the final artifact - such as debug
        // files, ancillary file and lock files
        auto artifactContainer = ArtifactUtil::createArtifact(ArtifactDesc::make(
            ArtifactKind::Container,
            ArtifactPayload::Unknown,
            ArtifactStyle::Unknown));

        auto slice = SliceUtil::asSlice(artifactList);

        artifactContainer->setChildren(slice.data, slice.count);

        artifact->addAssociated(artifactContainer);
    }

    *outArtifact = artifact.detach();
    return SLANG_OK;
}

} // namespace Slang
