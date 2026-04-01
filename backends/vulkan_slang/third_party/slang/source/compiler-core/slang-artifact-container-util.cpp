// slang-artifact-container-util.cpp
#include "slang-artifact-container-util.h"

#include "../core/slang-castable.h"
#include "../core/slang-file-system.h"
#include "../core/slang-io.h"
#include "../core/slang-string-slice-pool.h"
#include "../core/slang-zip-file-system.h"
#include "slang-artifact-desc-util.h"
#include "slang-artifact-representation-impl.h"
#include "slang-artifact-util.h"

namespace Slang
{

/*
Artifact file structure
=======================

There is many ways this could work, with different trade offs. The approach taken here is
to make *every* artifact be a directory. There are two special case directories "associated" and
"children", which hold the artifacts associated/chidren artifacts.

So for example if we have

```
thing.spv
  associated
    diagnostics
```

It will become

```
thing.spv
associated/0/diagnostics.diag
```

```
somemodule
  associated
    diagnostics
    0a0a0a.map
    0b0b0b.map
```

Becomes

```
somemodule.slang-module
associated/0/diagnostics
associated/0a0a0a/0a0a0a.map
associated/0b0b0b/0b0b0b.map
```

That is a little verbose, but if the associated artifacts have children/associated, then things
still work.

```
container
  a.spv
    associated
        diagnostics
  b.dxil
    associated
        diagnostics
        sourcemap
```

```
a/a.spv
a/associated/0/diagnostics.diagnostics
b/b.spv
b/associated/0/diagnostics.diagnostics
b/associated/1/sourcemap.map
```

*/

struct ArtifactContainerWriter
{
    struct Entry
    {
        String path;
        Index uniqueIndex = 0;
    };

    struct Scope
    {
        SlangResult pushAndRequireDirectory(ArtifactContainerWriter* writer, const String& name)
        {
            SLANG_ASSERT(writer);
            SLANG_ASSERT(m_writer == nullptr);

            SlangResult res = writer->pushAndRequireDirectory(name);

            if (SLANG_SUCCEEDED(res))
            {
                m_writer = writer;
            }

            return res;
        }

        ~Scope()
        {
            if (m_writer)
            {
                m_writer->pop();
            }
        }

        ArtifactContainerWriter* m_writer = nullptr;
    };

    void push(const String& name)
    {
        auto path = Path::combine(m_entry.path, name);

        m_entryStack.add(m_entry);

        Entry entry;
        entry.path = path;

        m_entry = entry;
    }
    SlangResult pushAndRequireDirectory(const String& name)
    {
        push(name);

        const char* const path = m_entry.path.getBuffer();

        SlangPathType pathType;
        if (SLANG_SUCCEEDED(m_fileSystem->getPathType(path, &pathType)))
        {
            if (pathType != SLANG_PATH_TYPE_DIRECTORY)
            {
                return SLANG_FAIL;
            }
        }

        // Make sure there is a path to this
        return m_fileSystem->createDirectory(m_entry.path.getBuffer());
    }
    void pop()
    {
        SLANG_ASSERT(m_entryStack.getCount() > 0);
        m_entry = m_entryStack.getLast();
        m_entryStack.removeLast();
    }

    SlangResult getBaseName(IArtifact* artifact, String& out);

    /// Write the artifact in the current scope
    SlangResult write(IArtifact* artifact);
    SlangResult writeInDirectory(IArtifact* artifact, const String& baseName);

    ArtifactContainerWriter(ISlangMutableFileSystem* fileSystem)
        : m_fileSystem(fileSystem)
    {
    }

    List<Entry> m_entryStack;
    Entry m_entry;

    ISlangMutableFileSystem* m_fileSystem;
};

SlangResult ArtifactContainerWriter::getBaseName(IArtifact* artifact, String& out)
{
    String baseName;

    const auto artifactDesc = artifact->getDesc();

    {
        auto artifactName = artifact->getName();
        if (artifactName && artifactName[0] != 0)
        {
            baseName = ArtifactDescUtil::getBaseNameFromPath(
                artifactDesc,
                UnownedStringSlice(artifactName));
        }
    }

    // If we don't have name, use a generated one
    if (baseName.getLength() == 0)
    {
        baseName.append(m_entry.uniqueIndex++);
    }

    out = baseName;
    return SLANG_OK;
}

SlangResult ArtifactContainerWriter::writeInDirectory(IArtifact* artifact, const String& baseName)
{
    // TODO(JS):
    // We could now output information about the desc/artifact, say as some json.
    // For now we assume the extension is good enough for most purposes.

    // If it's an "arbitrary" container, we don't need to write it
    if (artifact->getDesc().kind != ArtifactKind::Container)
    {
        // We can't write it without a blob
        ComPtr<ISlangBlob> blob;
        SLANG_RETURN_ON_FAIL(artifact->loadBlob(ArtifactKeep::No, blob.writeRef()));

        // Get the name of the artifact
        StringBuilder artifactName;
        SLANG_RETURN_ON_FAIL(ArtifactDescUtil::calcNameForDesc(
            artifact->getDesc(),
            baseName.getUnownedSlice(),
            artifactName));

        const auto combinedPath = Path::combine(m_entry.path, artifactName);
        // Write out the blob
        SLANG_RETURN_ON_FAIL(m_fileSystem->saveFileBlob(combinedPath.getBuffer(), blob));
    }

    {
        auto children = artifact->getChildren();
        if (children.count)
        {
            Scope childrenScope;
            SLANG_RETURN_ON_FAIL(childrenScope.pushAndRequireDirectory(this, "children"));

            for (IArtifact* child : children)
            {
                SLANG_RETURN_ON_FAIL(write(child));
            }
        }
    }
    {
        auto associatedSlice = artifact->getAssociated();
        if (associatedSlice.count)
        {
            Scope associatedScope;
            SLANG_RETURN_ON_FAIL(associatedScope.pushAndRequireDirectory(this, "associated"));

            for (IArtifact* associated : associatedSlice)
            {
                SLANG_RETURN_ON_FAIL(write(associated));
            }
        }
    }

    return SLANG_OK;
}

SlangResult ArtifactContainerWriter::write(IArtifact* artifact)
{
    String baseName;
    SLANG_RETURN_ON_FAIL(getBaseName(artifact, baseName));

    // We don't special case if the artifact contains no children/associated.
    // We always create a directory for all artifacts. This makes it more verbose,
    // but simplifies things, because *generally* an artifact including it's children/associated
    // meta data is all contained in a single directory

    {
        Scope artifactScope;
        SLANG_RETURN_ON_FAIL(artifactScope.pushAndRequireDirectory(this, baseName));
        SLANG_RETURN_ON_FAIL(writeInDirectory(artifact, baseName));
    }

    return SLANG_OK;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ArtifactContainerParser !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

struct FileSystemContents
{
    // The first entry is always the root path

    struct IndexRange
    {
        SLANG_FORCE_INLINE Index getCount() const { return endIndex - startIndex; }

        SLANG_FORCE_INLINE Index begin() const { return startIndex; }
        SLANG_FORCE_INLINE Index end() const { return endIndex; }

        void set(Index inStart, Index inEnd)
        {
            startIndex = inStart;
            endIndex = inEnd;
        }

        static IndexRange make(Index inStart, Index inEnd)
        {
            IndexRange range;
            range.set(inStart, inEnd);
            return range;
        }

        Index startIndex;
        Index endIndex;
    };

    struct Entry
    {
        bool isFile() const { return range.startIndex < 0; }
        bool isDirectory() const { return range.startIndex >= 0; }

        void setDirectory() { range.set(0, 0); }
        void setFile() { range.set(-1, -1); }

        void setType(SlangPathType type)
        {
            (type == SLANG_PATH_TYPE_FILE) ? setFile() : setDirectory();
        }

        void setDirectoryRange(Index inStartIndex, Index inEndIndex)
        {
            SLANG_ASSERT(inEndIndex >= inStartIndex);
            SLANG_ASSERT(isDirectory());
            range.set(inStartIndex, inEndIndex);
        }

        Index parentDirectoryIndex = -1; ///< The directory this entry is in. -1 is root.
        UnownedStringSlice name;         ///< Name of this entry
        IndexRange range = IndexRange::make(-1, -1); ///< Default to file
    };

    void clear()
    {
        m_pool.clear();
        m_entries.clear();
    }

    IndexRange getContentsRange(Index index) const { return m_entries[index].range; }

    ConstArrayView<Entry> getContents(Index index) const { return getContents(m_entries[index]); }
    ConstArrayView<Entry> getContents(const Entry& entry) const
    {
        return entry.range.getCount() ? makeConstArrayView(
                                            m_entries.getBuffer() + entry.range.startIndex,
                                            entry.range.getCount())
                                      : makeConstArrayView<Entry>(nullptr, 0);
    }

    void appendPath(Index entryIndex, StringBuilder& buf);

    SlangResult find(ISlangFileSystemExt* fileSyste, const UnownedStringSlice& path);

    FileSystemContents()
        : m_pool(StringSlicePool::Style::Default)
    {
        clear();
    }

    static void _add(SlangPathType pathType, const char* name, void* userData)
    {
        FileSystemContents* contents = (FileSystemContents*)userData;

        Entry entry;

        entry.parentDirectoryIndex = contents->m_currentParent;
        entry.name = contents->m_pool.addAndGetSlice(name);
        entry.setType(pathType);

        contents->m_entries.add(entry);
    }

    Index m_currentParent = -1; ///< Convenience for adding entries when using enumerate

    StringSlicePool m_pool; ///< Holds strings
    List<Entry> m_entries;  ///< The entries
};

void FileSystemContents::appendPath(Index entryIndex, StringBuilder& buf)
{
    const auto& entry = m_entries[entryIndex];
    if (entry.parentDirectoryIndex >= 0)
    {
        // If there is a parent recurse to append that first
        appendPath(entry.parentDirectoryIndex, buf);
    }

    // If the buffer is non zero, we need to add a separator
    if (buf.getLength() > 0)
    {
        buf.appendChar('/');
    }

    buf.append(entry.name);
}

SlangResult FileSystemContents::find(
    ISlangFileSystemExt* fileSystem,
    const UnownedStringSlice& inPath)
{
    clear();

    StringBuilder currentPath;
    currentPath.append(inPath);

    // If there is no name, just go with .
    const char* checkPath = currentPath.getLength() ? currentPath.getBuffer() : ".";

    SlangPathType pathType;
    SLANG_RETURN_ON_FAIL(fileSystem->getPathType(checkPath, &pathType));

    if (pathType == SLANG_PATH_TYPE_FILE)
    {
        Entry directoryEntry;
        directoryEntry.parentDirectoryIndex = -1;
        directoryEntry.name = m_pool.addAndGetSlice(Path::getParentDirectory(inPath));
        directoryEntry.range.set(1, 2);
        SLANG_ASSERT(directoryEntry.isDirectory());

        m_entries.add(directoryEntry);

        Entry entry;
        entry.parentDirectoryIndex = 0;
        entry.name = m_pool.addAndGetSlice(Path::getFileName(inPath));
        SLANG_ASSERT(entry.isFile());

        m_entries.add(entry);
    }
    else
    {
        Entry directoryEntry;
        directoryEntry.setDirectory();

        directoryEntry.name = m_pool.addAndGetSlice(inPath);
        m_entries.add(directoryEntry);

        for (Index i = 0; i < m_entries.getCount(); ++i)
        {
            const Entry entry = m_entries[i];

            if (entry.isDirectory())
            {
                // Clear the current path
                currentPath.clear();

                appendPath(i, currentPath);

                // Makes all the items added have this set as their parent
                m_currentParent = i;

                const auto startIndex = m_entries.getCount();

                const char* const path = currentPath.getLength() ? currentPath.getBuffer() : ".";

                const auto res = fileSystem->enumeratePathContents(path, _add, this);

                m_entries[i].setDirectoryRange(startIndex, m_entries.getCount());

                SLANG_RETURN_ON_FAIL(res);
            }
        }
    }

    return SLANG_OK;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ArtifactContainerUtil !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* static */ SlangResult ArtifactContainerUtil::writeContainer(
    IArtifact* artifact,
    const String& defaultFileName,
    ISlangMutableFileSystem* fileSystem)
{
    ArtifactContainerWriter writer(fileSystem);

    String baseName;

    {
        const char* name = artifact->getName();
        if (name == nullptr || name[0] == 0)
        {
            // Try to get the name from the defaultFileName
            baseName = Path::getFileNameWithoutExt(defaultFileName);
        }
    }

    // If it's still not set try generating it.
    if (baseName.getLength() == 0)
    {
        SLANG_RETURN_ON_FAIL(writer.getBaseName(artifact, baseName));
    }

    SLANG_RETURN_ON_FAIL(writer.writeInDirectory(artifact, baseName));

    return SLANG_OK;
}

static SlangResult _remove(ISlangMutableFileSystem* fileSystem, const String& path)
{
    SlangPathType pathType;
    if (SLANG_SUCCEEDED(fileSystem->getPathType(path.getBuffer(), &pathType)))
    {
        fileSystem->remove(path.getBuffer());
    }
    return SLANG_OK;
}

/* static */ SlangResult ArtifactContainerUtil::writeContainer(
    IArtifact* artifact,
    const String& fileName)
{
    auto osFileSystem = OSFileSystem::getMutableSingleton();

    const auto ext = Path::getPathExt(fileName);

    if (ext == toSlice("zip"))
    {
        SLANG_RETURN_ON_FAIL(_remove(osFileSystem, fileName));

        // Create the zip
        ComPtr<ISlangMutableFileSystem> fileSystem;
        SLANG_RETURN_ON_FAIL(ZipFileSystem::create(fileSystem));

        // Write everything out
        SLANG_RETURN_ON_FAIL(writeContainer(artifact, fileName, fileSystem));

        // Now write out to the output file
        IArchiveFileSystem* archiveFileSystem = as<IArchiveFileSystem>(fileSystem);
        SLANG_ASSERT(archiveFileSystem);

        ComPtr<ISlangBlob> blob;
        SLANG_RETURN_ON_FAIL(archiveFileSystem->storeArchive(false, blob.writeRef()));

        // Okay we can now write out the zip
        SLANG_RETURN_ON_FAIL(osFileSystem->saveFileBlob(fileName.getBuffer(), blob));
        return SLANG_OK;
    }
    else if (ext == toSlice("dir"))
    {
        // We use the special extension "dir" to write out to a directory.
        // This is a little hokey arguably...
        auto path = Path::getPathWithoutExt(fileName);

        SLANG_RETURN_ON_FAIL(_remove(osFileSystem, path));

        SLANG_RETURN_ON_FAIL(osFileSystem->createDirectory(path.getBuffer()));

        ComPtr<ISlangMutableFileSystem> fileSystem(new RelativeFileSystem(osFileSystem, path));

        SLANG_RETURN_ON_FAIL(writeContainer(artifact, fileName, fileSystem));
        return SLANG_OK;
    }

    // In order to write out as a artifact hierarchy we need a file system. If we don't have that
    // we only write out the "main" (or root) artifact. All associated/children are typically
    // ignored.
    {
        // Get the artifact as a blob
        ComPtr<ISlangBlob> containerBlob;
        SLANG_RETURN_ON_FAIL(artifact->loadBlob(ArtifactKeep::Yes, containerBlob.writeRef()));

        // Write out the blob
        SLANG_RETURN_ON_FAIL(osFileSystem->saveFileBlob(fileName.getBuffer(), containerBlob));
    }

    return SLANG_OK;
}

struct ArtifactContainerReader
{
    SlangResult read(ISlangFileSystemExt* fileSystem, ComPtr<IArtifact>& outArtifact);

    /// A directory that contains multiple artifact directories
    SlangResult _readContainerDirectory(
        Index directoryIndex,
        IArtifact::ContainedKind kind,
        IArtifact* container);
    /// A directory that holds a single
    SlangResult _readArtifactDirectory(Index directoryIndex, ComPtr<IArtifact>& outArtifact);

    SlangResult _readFile(Index fileIndex, ComPtr<IArtifact>& outArtifact);

    FileSystemContents m_contents;
    ISlangFileSystemExt* m_fileSystem;
};

SlangResult ArtifactContainerReader::read(
    ISlangFileSystemExt* fileSystem,
    ComPtr<IArtifact>& outArtifact)
{
    m_fileSystem = fileSystem;
    m_contents.find(fileSystem, toSlice(""));

    return _readArtifactDirectory(0, outArtifact);
}

SlangResult ArtifactContainerReader::_readFile(Index fileIndex, ComPtr<IArtifact>& outArtifact)
{
    outArtifact.setNull();

    const auto& entry = m_contents.m_entries[fileIndex];
    SLANG_ASSERT(entry.isFile());

    ArtifactDesc desc;

    auto ext = Path::getPathExt(entry.name);
    if (ext.getLength() == 0)
    {
        // I guess we'll assume it's an executable for now. We should use some kind of associated
        // information/manifest probly
        desc = ArtifactDesc::make(ArtifactKind::Executable, ArtifactPayload::HostCPU);
    }
    else
    {
        desc = ArtifactDescUtil::getDescFromPath(entry.name);
    }

    // Don't know what this is.
    if (desc.kind == ArtifactKind::Unknown || desc.kind == ArtifactKind::Invalid)
    {
        return SLANG_OK;
    }

    // We don't have manifest, so for now well assume if the name ends in "-obfuscated" and it's a
    // source map it's an obfuscated one
    if (desc.kind == ArtifactKind::Json && desc.payload == ArtifactPayload::SourceMap)
    {
        auto name = Path::getFileNameWithoutExt(entry.name);

        if (name.endsWith(toSlice("-obfuscated")))
        {
            desc.style = ArtifactStyle::Obfuscated;
        }
    }

    // I guess I can just make an artifact for this
    auto artifact = ArtifactUtil::createArtifact(desc);

    if (entry.name.getLength())
    {
        // We can set the name on the artifact if set
        // We know it's 0 terminated, because all names are in the pool
        // and therefore have to have 0 termination
        artifact->setName(entry.name.begin());
    }

    StringBuilder path;
    m_contents.appendPath(fileIndex, path);

    IExtFileArtifactRepresentation* rep =
        new ExtFileArtifactRepresentation(path.getUnownedSlice(), m_fileSystem);
    artifact->addRepresentation(rep);

    outArtifact = artifact;
    return SLANG_OK;
}

SlangResult ArtifactContainerReader::_readContainerDirectory(
    Index directoryIndex,
    IArtifact::ContainedKind kind,
    IArtifact* containerArtifact)
{
    // This directory only contains other directories which are artifacts
    // Files are ignored

    auto indexRange = m_contents.getContentsRange(directoryIndex);

    for (Index i = indexRange.startIndex; i < indexRange.endIndex; ++i)
    {
        const auto& entry = m_contents.m_entries[i];

        // We ignore files
        if (entry.isFile())
        {
            continue;
        }

        ComPtr<IArtifact> artifact;

        SLANG_RETURN_ON_FAIL(_readArtifactDirectory(i, artifact));

        if (artifact)
        {
            switch (kind)
            {
            case IArtifact::ContainedKind::Associated:
                containerArtifact->addAssociated(artifact);
                break;
            case IArtifact::ContainedKind::Children:
                containerArtifact->addChild(artifact);
                break;
            default:
                SLANG_ASSERT(!"Can't add artifact to this kind");
                return SLANG_FAIL;
            }
        }
    }

    return SLANG_OK;
}

SlangResult ArtifactContainerReader::_readArtifactDirectory(
    Index directoryIndex,
    ComPtr<IArtifact>& outArtifact)
{
    auto indexRange = m_contents.getContentsRange(directoryIndex);

    Index childrenIndex = -1;
    Index associatedIndex = -1;

    ComPtr<IArtifact> artifact;

    // Look for files
    for (Index i = indexRange.startIndex; i < indexRange.endIndex; ++i)
    {
        const auto& entry = m_contents.m_entries[i];
        if (entry.isFile())
        {
            ComPtr<IArtifact> readArtifact;
            SLANG_RETURN_ON_FAIL(_readFile(i, readArtifact));

            if (readArtifact)
            {
                if (artifact)
                {
                    // We can only have one artifact in the directory
                    return SLANG_FAIL;
                }
                artifact = readArtifact;
            }
        }
        else if (entry.isDirectory())
        {
            if (entry.name == toSlice("associated"))
            {
                associatedIndex = i;
            }
            else if (entry.name == toSlice("children"))
            {
                childrenIndex = i;
            }
        }
    }

    // If we didn't find an artifact so far
    if (!artifact)
    {
        // If we have children/associated we can assume it's a container
        if (childrenIndex >= 0 || associatedIndex >= 0)
        {
            artifact = ArtifactUtil::createArtifact(
                ArtifactDesc::make(ArtifactKind::Container, ArtifactPayload::Unknown));
            artifact->setName(m_contents.m_entries[directoryIndex].name.begin());
        }
        else
        {
            // Didn't find anything
            return SLANG_OK;
        }
    }

    if (childrenIndex >= 0)
    {
        SLANG_RETURN_ON_FAIL(
            _readContainerDirectory(childrenIndex, IArtifact::ContainedKind::Children, artifact));
    }
    if (associatedIndex >= 0)
    {
        SLANG_RETURN_ON_FAIL(_readContainerDirectory(
            associatedIndex,
            IArtifact::ContainedKind::Associated,
            artifact));
    }

    outArtifact = artifact;
    return SLANG_OK;
}

SlangResult ArtifactContainerUtil::readContainer(
    IArtifact* artifact,
    ComPtr<IArtifact>& outArtifact)
{
    auto desc = artifact->getDesc();

    ComPtr<ISlangMutableFileSystem> fileSystem;

    switch (desc.kind)
    {
    case ArtifactKind::Zip:
        {
            SLANG_RETURN_ON_FAIL(ZipFileSystem::create(fileSystem));

            ComPtr<ISlangBlob> blob;
            SLANG_RETURN_ON_FAIL(artifact->loadBlob(ArtifactKeep::No, blob.writeRef()));

            // Load into the zip

            // Now write out to the output file
            IArchiveFileSystem* archiveFileSystem = as<IArchiveFileSystem>(fileSystem);
            SLANG_ASSERT(archiveFileSystem);

            SLANG_RETURN_ON_FAIL(
                archiveFileSystem->loadArchive(blob->getBufferPointer(), blob->getBufferSize()));
            break;
        }
    default:
        {
            return SLANG_FAIL;
        }
    }

    SLANG_RETURN_ON_FAIL(readContainer(fileSystem, outArtifact));
    return SLANG_OK;
}

/* static */ SlangResult ArtifactContainerUtil::readContainer(
    ISlangFileSystemExt* fileSystem,
    ComPtr<IArtifact>& outArtifact)
{
    SLANG_UNUSED(outArtifact);

    ArtifactContainerReader reader;
    SLANG_RETURN_ON_FAIL(reader.read(fileSystem, outArtifact));

    return SLANG_OK;
}

/* static */ SlangResult ArtifactContainerUtil::filter(
    IArtifact* artifact,
    ComPtr<IArtifact>& outArtifact)
{
    outArtifact.setNull();

    // Copy the artifact
    auto dstArtifact = ArtifactUtil::createArtifact(artifact->getDesc(), artifact->getName());

    ComPtr<ISlangBlob> blob;

    if (artifact->getDesc().kind != ArtifactKind::Container)
    {
        // We can't write it without a blob
        const auto res = artifact->loadBlob(ArtifactKeep::No, blob.writeRef());

        if (SLANG_FAILED(res))
        {
            // If it failed and it's significant the whole write fails
            if (ArtifactUtil::isSignificant(artifact))
            {
                return res;
            }
        }
        else
        {
            // Add the blob to the destination
            dstArtifact->addRepresentationUnknown(blob);
        }
    }

    // Copy the children after filtering
    {
        for (IArtifact* child : artifact->getChildren())
        {
            ComPtr<IArtifact> dstChild;
            SLANG_RETURN_ON_FAIL(filter(child, dstChild));

            if (dstChild)
            {
                dstArtifact->addChild(dstChild);
            }
        }
    }

    // Copy the associated after filtering
    {
        for (IArtifact* assoc : artifact->getAssociated())
        {
            ComPtr<IArtifact> dstAssoc;
            SLANG_RETURN_ON_FAIL(filter(assoc, dstAssoc));

            if (dstAssoc)
            {
                dstArtifact->addAssociated(dstAssoc);
            }
        }
    }

    // We only return the artifact if any of the following are true
    // 1) It has a blob representation
    // 2) It contains children or associated artifacts

    if (blob || dstArtifact->getChildren().count || dstArtifact->getAssociated().count)
    {
        outArtifact = dstArtifact;
    }

    // If we return an artifact or not, this was successful
    return SLANG_OK;
}

} // namespace Slang
