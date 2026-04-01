#include "slang-memory-file-system.h"

// For Path::
#include "slang-blob.h"
#include "slang-implicit-directory-collector.h"
#include "slang-io.h"

namespace Slang
{

MemoryFileSystem::MemoryFileSystem()
{
    m_rootEntry.initDirectory("/");
}

void* MemoryFileSystem::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ISlangCastable::getTypeGuid() ||
        guid == ISlangFileSystem::getTypeGuid() || guid == ISlangFileSystemExt::getTypeGuid() ||
        guid == ISlangMutableFileSystem::getTypeGuid())
    {
        return static_cast<ISlangMutableFileSystem*>(this);
    }
    return nullptr;
}

void* MemoryFileSystem::getObject(const Guid& guid)
{
    SLANG_UNUSED(guid);
    return nullptr;
}

void* MemoryFileSystem::castAs(const Guid& guid)
{
    if (auto ptr = getInterface(guid))
    {
        return ptr;
    }
    return getObject(guid);
}

void MemoryFileSystem::_clear()
{
    m_entries = Dictionary<String, Entry>();
}

MemoryFileSystem::Entry* MemoryFileSystem::_getEntryFromCanonicalPath(const String& canonicalPath)
{
    if (canonicalPath == toSlice("."))
    {
        return &m_rootEntry;
    }
    else
    {
        return m_entries.tryGetValue(canonicalPath);
    }
}

MemoryFileSystem::Entry* MemoryFileSystem::_getEntryFromPath(const char* path, String* outPath)
{
    StringBuilder buffer;
    if (SLANG_FAILED(_getCanonical(path, buffer)))
    {
        return nullptr;
    }

    if (outPath)
    {
        *outPath = buffer;
    }
    return _getEntryFromCanonicalPath(buffer);
}

SlangResult MemoryFileSystem::_loadFile(const char* path, Entry** outEntry)
{
    *outEntry = nullptr;
    Entry* entry = _getEntryFromPath(path);
    if (entry == nullptr || entry->m_type != SLANG_PATH_TYPE_FILE)
    {
        return SLANG_E_NOT_FOUND;
    }
    *outEntry = entry;
    return SLANG_OK;
}

SlangResult MemoryFileSystem::loadFile(char const* path, ISlangBlob** outBlob)
{
    Entry* entry;
    SLANG_RETURN_ON_FAIL(_loadFile(path, &entry));

    ISlangBlob* contents = entry->m_contents;
    contents->addRef();
    *outBlob = contents;

    return SLANG_OK;
}

SlangResult MemoryFileSystem::getFileUniqueIdentity(
    const char* path,
    ISlangBlob** outUniqueIdentity)
{
    return getPath(PathKind::Canonical, path, outUniqueIdentity);
}

SlangResult MemoryFileSystem::calcCombinedPath(
    SlangPathType fromPathType,
    const char* fromPath,
    const char* path,
    ISlangBlob** pathOut)
{
    String combinedPath;
    switch (fromPathType)
    {
    case SLANG_PATH_TYPE_FILE:
        {
            combinedPath = Path::combine(Path::getParentDirectory(fromPath), path);
            break;
        }
    case SLANG_PATH_TYPE_DIRECTORY:
        {
            combinedPath = Path::combine(fromPath, path);
            break;
        }
    }

    *pathOut = StringBlob::moveCreate(combinedPath).detach();
    return SLANG_OK;
}

SlangResult MemoryFileSystem::getPathType(const char* path, SlangPathType* outPathType)
{
    if (auto entry = _getEntryFromPath(path))
    {
        *outPathType = entry->m_type;
        return SLANG_OK;
    }
    // Not found
    return SLANG_E_NOT_FOUND;
}

SlangResult MemoryFileSystem::getPath(PathKind kind, const char* path, ISlangBlob** outPath)
{
    switch (kind)
    {
    case PathKind::Simplified:
        {
            String simplifiedPath = Path::simplify(path);
            *outPath = StringBlob::moveCreate(simplifiedPath).detach();
            return SLANG_OK;
        }
    case PathKind::Display:
    case PathKind::Canonical:
        {
            StringBuilder buffer;
            SLANG_RETURN_ON_FAIL(_getCanonical(path, buffer));
            *outPath = StringBlob::moveCreate(buffer).detach();
            return SLANG_OK;
        }
    default:
        break;
    }
    return SLANG_E_NOT_AVAILABLE;
}

SlangResult MemoryFileSystem::enumeratePathContents(
    const char* path,
    FileSystemContentsCallBack callback,
    void* userData)
{
    String canonicalPath;
    Entry* entry = _getEntryFromPath(path, &canonicalPath);

    if (!entry || entry->m_type != SLANG_PATH_TYPE_DIRECTORY)
    {
        return SLANG_E_NOT_FOUND;
    }

    ImplicitDirectoryCollector collector(canonicalPath, true);

    // If it is a directory, we need to see if there is anything in it
    for (const auto& [_, childEntry] : m_entries)
    {
        collector.addPath(childEntry.m_type, childEntry.m_canonicalPath.getUnownedSlice());
    }

    return collector.enumerate(callback, userData);
}

SlangResult MemoryFileSystem::saveFile(const char* path, const void* data, size_t size)
{
    Entry* entry;
    SLANG_RETURN_ON_FAIL(_requireFile(path, &entry));
    auto contents = RawBlob::create(data, size);
    entry->setContents(size, contents);
    return SLANG_OK;
}

SlangResult MemoryFileSystem::saveFileBlob(const char* path, ISlangBlob* dataBlob)
{
    if (!dataBlob)
    {
        return SLANG_E_INVALID_ARG;
    }

    Entry* entry;
    SLANG_RETURN_ON_FAIL(_requireFile(path, &entry));
    entry->setContents(dataBlob->getBufferSize(), dataBlob);
    return SLANG_OK;
}

SlangResult MemoryFileSystem::_getCanonical(const char* path, StringBuilder& outCanonicalPath)
{
    StringBuilder canonicalPath;
    SLANG_RETURN_ON_FAIL(Path::simplify(
        UnownedStringSlice(path),
        Path::SimplifyStyle::AbsoluteOnlyAndNoRoot,
        outCanonicalPath));
    return SLANG_OK;
}

SlangResult MemoryFileSystem::_getCanonicalWithExistingParent(
    const char* path,
    StringBuilder& outCanonicalPath)
{
    SLANG_RETURN_ON_FAIL(_getCanonical(path, outCanonicalPath));

    // Get the parent to the canoncial path (which should be canonical itself)
    auto parent = Path::getParentDirectory(outCanonicalPath);

    if (parent.getLength())
    {
        // The parent has to be a directory
        Entry* parentEntry = _getEntryFromCanonicalPath(parent);
        if (parentEntry == nullptr || parentEntry->m_type != SLANG_PATH_TYPE_DIRECTORY)
        {
            return SLANG_E_NOT_FOUND;
        }
    }

    return SLANG_OK;
}

SlangResult MemoryFileSystem::_requireFile(const char* path, Entry** outEntry)
{
    *outEntry = nullptr;

    StringBuilder canonicalPath;
    SLANG_RETURN_ON_FAIL(_getCanonicalWithExistingParent(path, canonicalPath));

    Entry* foundEntry = _getEntryFromCanonicalPath(canonicalPath);

    if (foundEntry)
    {
        if (foundEntry->m_type != SLANG_PATH_TYPE_FILE)
        {
            // Can only set if it's already a file, if it's anything else it's an error
            return SLANG_FAIL;
        }
    }
    else
    {
        Entry entry;
        entry.initFile(canonicalPath);
        m_entries.add(canonicalPath, entry);

        foundEntry = _getEntryFromCanonicalPath(canonicalPath);
    }

    // It must be found and be a file
    SLANG_ASSERT(
        foundEntry && foundEntry->m_type == SLANG_PATH_TYPE_FILE &&
        foundEntry->m_canonicalPath == canonicalPath);

    *outEntry = foundEntry;
    return SLANG_OK;
}

SlangResult MemoryFileSystem::remove(const char* path)
{
    String canonicalPath;
    Entry* entry = _getEntryFromPath(path, &canonicalPath);

    // If there is an entry and not the root of the file system
    if (entry && entry != &m_rootEntry)
    {
        if (entry->m_type == SLANG_PATH_TYPE_DIRECTORY)
        {
            ImplicitDirectoryCollector collector(canonicalPath);

            // If it is a directory, we need to see if there is anything in it
            for (const auto& [_, childEntry] : m_entries)
            {
                collector.addPath(childEntry.m_type, childEntry.m_canonicalPath.getUnownedSlice());
                if (collector.hasContent())
                {
                    // Directory is not empty
                    return SLANG_FAIL;
                }
            }
        }

        // Reset so doesn't hold references/keep memory in scope
        entry->reset();
        m_entries.remove(canonicalPath);
        return SLANG_OK;
    }

    return SLANG_E_NOT_FOUND;
}

SlangResult MemoryFileSystem::createDirectory(const char* path)
{
    StringBuilder canonicalPath;
    SLANG_RETURN_ON_FAIL(_getCanonicalWithExistingParent(path, canonicalPath));

    if (_getEntryFromCanonicalPath(canonicalPath))
    {
        return SLANG_FAIL;
    }

    Entry entry;
    entry.initDirectory(canonicalPath);
    m_entries.add(canonicalPath, entry);
    return SLANG_OK;
}

} // namespace Slang
