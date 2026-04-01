// unit-test-io.cpp

#include "../../source/core/slang-io.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

static SlangResult _checkGenerateTemporary()
{
    /// Test temporary file functionality

    List<String> paths;

    for (Index i = 0; i < 10; ++i)
    {
        String path;
        SLANG_RETURN_ON_FAIL(File::generateTemporary(toSlice("slang-check"), path));

        // The path should exist exist
        SLANG_CHECK(File::exists(path));

        if (paths.contains(path))
        {
            return SLANG_FAIL;
        }

        paths.add(path);
    }

    // It should be possible to write to the temporary files
    for (auto& path : paths)
    {
        SLANG_RETURN_ON_FAIL(File::writeAllText(path, path));
    }
    // It should be possible to read from the temporary files

    for (auto& path : paths)
    {
        String contents;
        SLANG_RETURN_ON_FAIL(File::readAllText(path, contents))

        SLANG_CHECK(contents == path);
    }

    // Remove all the temporary files
    for (auto& path : paths)
    {
        SLANG_CHECK(File::exists(path));

        const auto removeResult = File::remove(path);
        SLANG_CHECK(SLANG_SUCCEEDED(removeResult));

        // Check remove worked
        SLANG_CHECK(!File::exists(path));
    }

    return SLANG_OK;
}

SLANG_UNIT_TEST(io)
{
    SLANG_CHECK(SLANG_SUCCEEDED(_checkGenerateTemporary()));
}
