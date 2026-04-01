// unit-test-file-system.cpp

#include "../../source/core/slang-castable.h"
#include "../../source/core/slang-deflate-compression-system.h"
#include "../../source/core/slang-file-system.h"
#include "../../source/core/slang-io.h"
#include "../../source/core/slang-lz4-compression-system.h"
#include "../../source/core/slang-memory-file-system.h"
#include "../../source/core/slang-riff-file-system.h"
#include "../../source/core/slang-zip-file-system.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

namespace
{ // anonymous

enum class FileSystemType
{
    Zip,
    RiffUncompressed,
    RiffDeflate,
    RiffLZ4,
    Memory,
    Relative,
    CountOf,
};

struct Entry
{
    typedef Entry ThisType;

    bool operator<(const ThisType& rhs) const { return path < rhs.path; }
    bool operator==(const ThisType& rhs) const { return path == rhs.path && type == rhs.type; }
    bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }

    SlangPathType type;
    String path;
};

} // namespace

static SlangResult _checkFile(
    ISlangFileSystemExt* fileSystem,
    const char* path,
    const UnownedStringSlice& contentsSlice)
{
    SlangPathType pathType;
    SLANG_RETURN_ON_FAIL(fileSystem->getPathType(path, &pathType));

    if (pathType != SLANG_PATH_TYPE_FILE)
    {
        return SLANG_FAIL;
    }

    ComPtr<ISlangBlob> blob;
    SLANG_RETURN_ON_FAIL(fileSystem->loadFile(path, blob.writeRef()));

    if (blob->getBufferSize() != contentsSlice.getLength())
    {
        return SLANG_FAIL;
    }
    if (contentsSlice !=
        UnownedStringSlice((const char*)blob->getBufferPointer(), blob->getBufferSize()))
    {
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

static SlangResult _checkFile(
    ISlangMutableFileSystem* fileSystem,
    const char* path,
    const char* contents)
{
    return _checkFile(fileSystem, path, UnownedStringSlice(contents));
}

static SlangResult _checkDirectoryExists(ISlangFileSystemExt* fileSystem, const char* path)
{
    SlangPathType pathType;
    SLANG_RETURN_ON_FAIL(fileSystem->getPathType(path, &pathType));

    if (pathType != SLANG_PATH_TYPE_DIRECTORY)
    {
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

static SlangResult _createAndCheckFile(
    ISlangMutableFileSystem* fileSystem,
    const char* path,
    const char* contents)
{
    UnownedStringSlice contentsSlice(contents);

    SLANG_RETURN_ON_FAIL(
        fileSystem->saveFile(path, contentsSlice.begin(), contentsSlice.getLength()));
    SLANG_RETURN_ON_FAIL(_checkFile(fileSystem, path, contentsSlice));

    // Delete it
    SLANG_RETURN_ON_FAIL(fileSystem->remove(path));

    // Check it's gone
    SlangPathType pathType;
    if (SLANG_SUCCEEDED(fileSystem->getPathType(path, &pathType)))
    {
        return SLANG_FAIL;
    }

    // Save as a blob
    ComPtr<ISlangBlob> blob = RawBlob::create(contentsSlice.begin(), contentsSlice.getLength());

    SLANG_RETURN_ON_FAIL(fileSystem->saveFileBlob(path, blob));
    SLANG_RETURN_ON_FAIL(_checkFile(fileSystem, path, contentsSlice));

    return SLANG_OK;
}

static bool _areEqual(ISlangBlob* a, ISlangBlob* b)
{
    if (a == b)
    {
        return true;
    }
    if ((!a || !b) || (a->getBufferSize() != b->getBufferSize()))
    {
        return false;
    }

    return ::memcmp(a->getBufferPointer(), b->getBufferPointer(), a->getBufferSize()) == 0;
}

static SlangResult _checkCanonical(
    ISlangMutableFileSystem* fileSystem,
    const char* const* paths,
    Count count)
{
    if (count <= 0)
    {
        return SLANG_FAIL;
    }

    // The path has to exist to something for canonicalization to be relied upon
    SlangPathType pathType;
    SLANG_RETURN_ON_FAIL(fileSystem->getPathType(paths[0], &pathType));

    String canonicalPath;
    {
        ComPtr<ISlangBlob> blob;
        SLANG_RETURN_ON_FAIL(fileSystem->getPath(PathKind::Canonical, paths[0], blob.writeRef()));
        canonicalPath = StringUtil::getString(blob);
    }

    // The canonicalized path must point to the same thing
    SlangPathType canonicalPathType;
    SLANG_RETURN_ON_FAIL(fileSystem->getPathType(canonicalPath.getBuffer(), &canonicalPathType));

    if (canonicalPathType != pathType)
    {
        return SLANG_FAIL;
    }

    // If they are the file, being hte same file, they must hold the same data...
    if (pathType == SLANG_PATH_TYPE_FILE)
    {
        ComPtr<ISlangBlob> blob;
        ComPtr<ISlangBlob> canonicalPathBlob;
        SLANG_RETURN_ON_FAIL(fileSystem->loadFile(paths[0], blob.writeRef()));
        SLANG_RETURN_ON_FAIL(
            fileSystem->loadFile(canonicalPath.getBuffer(), canonicalPathBlob.writeRef()));

        if (!_areEqual(blob, canonicalPathBlob))
        {
            return SLANG_FAIL;
        }
    }

    for (Index i = 1; i < count; ++i)
    {
        ComPtr<ISlangBlob> blob;
        SLANG_RETURN_ON_FAIL(fileSystem->getPath(PathKind::Canonical, paths[i], blob.writeRef()));
        const auto checkPath = StringUtil::getString(blob);

        if (checkPath != canonicalPath)
        {
            return SLANG_FAIL;
        }
    }

    return SLANG_OK;
}


static SlangResult _createAndCheckDirectory(ISlangMutableFileSystem* fileSystem, const char* path)
{
    SLANG_RETURN_ON_FAIL(fileSystem->createDirectory(path));

    SlangPathType pathType;
    SLANG_RETURN_ON_FAIL(fileSystem->getPathType(path, &pathType));

    if (pathType != SLANG_PATH_TYPE_DIRECTORY)
    {
        return SLANG_FAIL;
    }

    return SLANG_OK;
}

static void _entryCallback(SlangPathType pathType, const char* name, void* userData)
{
    List<Entry>& out = *(List<Entry>*)userData;
    out.add(Entry{pathType, name});
}

static SlangResult _enumeratePath(
    ISlangFileSystemExt* fileSystem,
    const char* path,
    const ConstArrayView<Entry>& entries)
{
    List<Entry> contents;

    SLANG_RETURN_ON_FAIL(fileSystem->enumeratePathContents(path, _entryCallback, (void*)&contents));

    contents.sort();

    if (contents.getArrayView() != entries)
    {
        return SLANG_FAIL;
    }

    return SLANG_OK;
}

static SlangResult _checkSimplifiedPath(
    ISlangFileSystemExt* fileSystem,
    const char* path,
    const char* normalPath)
{
    ComPtr<ISlangBlob> simplifiedPathBlob;
    SLANG_RETURN_ON_FAIL(
        fileSystem->getPath(PathKind::Simplified, path, simplifiedPathBlob.writeRef()));

    auto simplifiedPath = StringUtil::getString(simplifiedPathBlob);

    if (simplifiedPath != normalPath)
    {
        return SLANG_FAIL;
    }

    return SLANG_OK;
}

SlangResult _appendPathEntries(
    ISlangFileSystemExt* fileSystem,
    const char* inBasePath,
    List<Entry>& outEntries)
{
    const UnownedStringSlice basePath(inBasePath);
    if (basePath == toSlice(".") || basePath.getLength() == 0)
    {
        // We don't need to append path prefixes if we are at the root.
        SLANG_RETURN_ON_FAIL(
            fileSystem->enumeratePathContents(inBasePath, _entryCallback, (void*)&outEntries));
    }
    else
    {
        const Index startIndex = outEntries.getCount();
        SLANG_RETURN_ON_FAIL(
            fileSystem->enumeratePathContents(inBasePath, _entryCallback, (void*)&outEntries));

        const String basePathString(basePath);

        // we need to fix all of the added paths to make absolute
        const Count count = outEntries.getCount();
        for (Index i = startIndex; i < count; ++i)
        {
            auto& entry = outEntries[i];
            entry.path = Path::combine(basePathString, entry.path);
        }
    }

    return SLANG_OK;
}

static SlangResult _getAllEntries(
    ISlangFileSystemExt* fileSystem,
    const char* inBasePath,
    List<Entry>& outEntries)
{
    outEntries.clear();

    // Simplify the base
    auto basePath = Path::simplify(inBasePath);

    _appendPathEntries(fileSystem, basePath.getBuffer(), outEntries);

    for (Index i = 0; i < outEntries.getCount(); ++i)
    {
        // We need to make a copy as outEntries is mutated
        const Entry entry = outEntries[i];
        if (entry.type == SLANG_PATH_TYPE_DIRECTORY)
        {
            _appendPathEntries(fileSystem, entry.path.getBuffer(), outEntries);
        }
    }

    // Sort to remove issues with traversal ordering
    outEntries.sort();
    return SLANG_OK;
}

static SlangResult _checkEqual(ISlangFileSystemExt* a, ISlangFileSystemExt* b)
{
    List<Entry> aEntries, bEntries;

    SLANG_RETURN_ON_FAIL(_getAllEntries(a, ".", aEntries));
    SLANG_RETURN_ON_FAIL(_getAllEntries(b, ".", bEntries));

    if (aEntries != bEntries)
    {
        return SLANG_FAIL;
    }

    // For all the files check the contents is the same

    for (const auto& entry : aEntries)
    {
        if (entry.type != SLANG_PATH_TYPE_FILE)
        {
            continue;
        }

        ComPtr<ISlangBlob> blobA, blobB;

        SLANG_RETURN_ON_FAIL(a->loadFile(entry.path.getBuffer(), blobA.writeRef()));
        SLANG_RETURN_ON_FAIL(b->loadFile(entry.path.getBuffer(), blobB.writeRef()));

        if (blobA->getBufferSize() != blobB->getBufferSize())
        {
            return SLANG_FAIL;
        }

        if (::memcmp(
                blobA->getBufferPointer(),
                blobB->getBufferPointer(),
                blobA->getBufferSize()) != 0)
        {
            return SLANG_FAIL;
        }
    }

    return SLANG_OK;
}

static SlangResult _createFileSystem(
    FileSystemType type,
    ComPtr<ISlangMutableFileSystem>& outFileSystem)
{
    outFileSystem.setNull();
    switch (type)
    {
    case FileSystemType::Zip:
        return ZipFileSystem::create(outFileSystem);
    case FileSystemType::RiffUncompressed:
        outFileSystem = new RiffFileSystem(nullptr);
        break;
    case FileSystemType::RiffDeflate:
        outFileSystem = new RiffFileSystem(DeflateCompressionSystem::getSingleton());
        break;
    case FileSystemType::RiffLZ4:
        outFileSystem = new RiffFileSystem(LZ4CompressionSystem::getSingleton());
        break;
    case FileSystemType::Memory:
        outFileSystem = new MemoryFileSystem;
        break;
    case FileSystemType::Relative:
        {
            ComPtr<ISlangMutableFileSystem> memoryFileSystem(new MemoryFileSystem);
            memoryFileSystem->createDirectory("base");

            outFileSystem = new RelativeFileSystem(memoryFileSystem, "base");
            break;
        }
    }

    return outFileSystem ? SLANG_OK : SLANG_FAIL;
}

static SlangResult _testImplicitDirectory(FileSystemType type)
{
    ComPtr<ISlangMutableFileSystem> fileSystem;
    SLANG_RETURN_ON_FAIL(_createFileSystem(type, fileSystem));

    const char contents3[] = "Some text....";

    SLANG_RETURN_ON_FAIL(
        fileSystem->saveFile("implicit-path/file2.txt", contents3, SLANG_COUNT_OF(contents3)));

    {
        SlangPathType pathType;
        SLANG_RETURN_ON_FAIL(fileSystem->getPathType("implicit-path", &pathType));

        SLANG_CHECK(pathType == SLANG_PATH_TYPE_DIRECTORY);

        auto checkEntries = [&]() -> SlangResult
        {
            List<Entry> entries;
            SLANG_RETURN_ON_FAIL(_getAllEntries(fileSystem, "implicit-path", entries));

            // It contains a file
            SLANG_CHECK(entries.getCount() == 1);

            for (const auto& entry : entries)
            {
                // All of these should exist
                SlangPathType pathType;
                SLANG_RETURN_ON_FAIL(fileSystem->getPathType(entry.path.getBuffer(), &pathType));
            }
            return SLANG_OK;
        };

        SLANG_RETURN_ON_FAIL(checkEntries());

        // Make an explicit path, and see whe have the same results
        fileSystem->createDirectory("implicit-path");

        SLANG_RETURN_ON_FAIL(checkEntries());
    }

    return SLANG_OK;
}

static SlangResult _test(FileSystemType type)
{
    ComPtr<ISlangMutableFileSystem> fileSystem;
    SLANG_RETURN_ON_FAIL(_createFileSystem(type, fileSystem));

    const auto aText = "someText";
    const auto bText = "A longer bit of text....";
    const auto d_aText = "Some more silly stuff";
    const auto d_bText = "Lets go!";

    SLANG_RETURN_ON_FAIL(_createAndCheckFile(fileSystem, "a", aText));
    SLANG_RETURN_ON_FAIL(_createAndCheckFile(fileSystem, "b", bText));

    SLANG_RETURN_ON_FAIL(_createAndCheckDirectory(fileSystem, "d"));
    SLANG_RETURN_ON_FAIL(_createAndCheckFile(fileSystem, "d/a", d_aText));
    SLANG_RETURN_ON_FAIL(_createAndCheckFile(fileSystem, "d\\b", d_bText));

    // Try and absolute path
    SLANG_RETURN_ON_FAIL(_checkFile(fileSystem, "/a", aText));
    SLANG_RETURN_ON_FAIL(_checkFile(fileSystem, "/b", bText));
    SLANG_RETURN_ON_FAIL(_checkFile(fileSystem, "/d/a", d_aText));
    SLANG_RETURN_ON_FAIL(_checkFile(fileSystem, "/d\\b", d_bText));


    // Check canonical on files
    {
        const char* paths[] = {"a", "/a", "./a", "d/../a", ".\\d/.\\..\\a"};
        SLANG_RETURN_ON_FAIL(_checkCanonical(fileSystem, paths, SLANG_COUNT_OF(paths)));
    }

    {
        const char* paths[] = {"/d/b", "d/./b"};
        SLANG_RETURN_ON_FAIL(_checkCanonical(fileSystem, paths, SLANG_COUNT_OF(paths)));
    }

    // Check canonical on directories
    {
        const char* paths[] = {".", "/", "/d/..", "d/.."};
        SLANG_RETURN_ON_FAIL(_checkCanonical(fileSystem, paths, SLANG_COUNT_OF(paths)));
    }

    {
        const char* paths[] = {"d", "./d", "/d", "/d/./../d"};
        SLANG_RETURN_ON_FAIL(_checkCanonical(fileSystem, paths, SLANG_COUNT_OF(paths)));
    }

    // Lets find all the files in the directory

    {
        const Entry entries[] = {{SLANG_PATH_TYPE_FILE, "a"}, {SLANG_PATH_TYPE_FILE, "b"}};
        SLANG_RETURN_ON_FAIL(_enumeratePath(fileSystem, "d", makeConstArrayView(entries)));
    }

    {
        const Entry entries[] = {
            {SLANG_PATH_TYPE_FILE, "a"},
            {SLANG_PATH_TYPE_FILE, "b"},
            {SLANG_PATH_TYPE_DIRECTORY, "d"}};
        SLANG_RETURN_ON_FAIL(_enumeratePath(fileSystem, ".", makeConstArrayView(entries)));

        // Let's check that / and \ works for the root directory
        SLANG_RETURN_ON_FAIL(_enumeratePath(fileSystem, "/", makeConstArrayView(entries)));
        SLANG_RETURN_ON_FAIL(_enumeratePath(fileSystem, "\\", makeConstArrayView(entries)));
    }

    // Check the root directory exists
    {
        SLANG_RETURN_ON_FAIL(_checkDirectoryExists(fileSystem, "."));
        SLANG_RETURN_ON_FAIL(_checkDirectoryExists(fileSystem, "/"));
        SLANG_RETURN_ON_FAIL(_checkDirectoryExists(fileSystem, "\\"));
    }

    {
        SLANG_RETURN_ON_FAIL(_checkSimplifiedPath(fileSystem, "d/../a", "a"));
    }


    // If we have an archive file system check out it's behavior
    if (IArchiveFileSystem* archiveFileSystem = as<IArchiveFileSystem>(fileSystem))
    {
        // Load and check its okay

        ComPtr<ISlangBlob> archiveBlob;
        SLANG_RETURN_ON_FAIL(archiveFileSystem->storeArchive(false, archiveBlob.writeRef()));

        ComPtr<ISlangFileSystemExt> loadedFileSystem;
        SLANG_RETURN_ON_FAIL(loadArchiveFileSystem(
            archiveBlob->getBufferPointer(),
            archiveBlob->getBufferSize(),
            loadedFileSystem));

        // Check the file systems contents are the same
        SLANG_RETURN_ON_FAIL(_checkEqual(loadedFileSystem, fileSystem));
    }

    SLANG_RETURN_ON_FAIL(fileSystem->remove("d/a"));
    {
        const Entry entries[] = {{SLANG_PATH_TYPE_FILE, "b"}};
        SLANG_RETURN_ON_FAIL(_enumeratePath(fileSystem, "d", makeConstArrayView(entries)));
    }
    SLANG_RETURN_ON_FAIL(fileSystem->remove("d\\b"));
    {
        SLANG_RETURN_ON_FAIL(
            _enumeratePath(fileSystem, "d", makeConstArrayView((const Entry*)nullptr, 0)));
    }

    // If it's removed it can't be removed again
    SLANG_CHECK(SLANG_FAILED(fileSystem->remove("d\\b")));

    // Remove the directory
    SLANG_RETURN_ON_FAIL(fileSystem->remove("d"));

    {
        const Entry entries[] = {{SLANG_PATH_TYPE_FILE, "a"}, {SLANG_PATH_TYPE_FILE, "b"}};
        SLANG_RETURN_ON_FAIL(_enumeratePath(fileSystem, ".", makeConstArrayView(entries)));
    }

    return SLANG_OK;
}

SLANG_UNIT_TEST(fileSystem)
{
    for (Index i = 0; i < Count(FileSystemType::CountOf); ++i)
    {
        const auto type = FileSystemType(i);

        SLANG_CHECK(SLANG_SUCCEEDED(_test(type)));

        // Some file system types support 'implicit directories'.
        // This means that if a file is created with a path, the directories
        // required to make that path valid are 'implicitly' created.
        //
        // Currently this behavior is supported by zip, and this test checks
        // that it is working correctly, as we require the file system to
        // behave correctly in other ways irrespectively of if the directory is
        // implicit or not.
        const bool hasImplicitDirectory = (type == FileSystemType::Zip);
        if (hasImplicitDirectory)
        {
            SLANG_CHECK(SLANG_SUCCEEDED(_testImplicitDirectory(type)));
        }
    }
}
