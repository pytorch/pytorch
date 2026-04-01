// slang-artifact-container-util.h
#ifndef SLANG_ARTIFACT_CONTAINER_UTIL_H
#define SLANG_ARTIFACT_CONTAINER_UTIL_H

#include "slang-artifact-representation.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"

namespace Slang
{

/* Functionality to save of and read artifact hierarchies via the slang
artifact container style. This style treats storage as a file system.

Since this represention, is not directly a file system representation
some conventions are used to associate data, via names etc.

The use of ISlangMutableFileSystem, allows writing this structure, to memory, zip, riff, directory
or other file like representations
*/
struct ArtifactContainerUtil
{
    /// Write the container using the specified path.
    /// Uses the extension of the path to determine how to write
    static SlangResult writeContainer(IArtifact* artifact, const String& path);

    /// If there isn't a suitable name on artifact, the filename is used to generate a name. If it's
    /// not set a name may be generated.
    static SlangResult writeContainer(
        IArtifact* artifact,
        const String& defaultFileName,
        ISlangMutableFileSystem* fileSystem);

    static SlangResult readContainer(
        ISlangFileSystemExt* fileSystem,
        ComPtr<IArtifact>& outArtifact);

    /// Read an artifact that represents a container as an artifact hierarchy
    static SlangResult readContainer(IArtifact* artifact, ComPtr<IArtifact>& outArtifact);

    /// Creates a copy of artifact where
    /// * All artifacts are blobs
    /// * Any generic containers that are empty are dropped
    /// * Any sub artifact that can't be blobed and isn't significant is ignored
    ///
    /// A future improvement would be to take a function to also control what makes it to the output
    static SlangResult filter(IArtifact* artifact, ComPtr<IArtifact>& outArtifact);
};

} // namespace Slang

#endif
