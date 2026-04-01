// unit-test-path.cpp

#include "../../source/core/slang-io.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

SLANG_UNIT_TEST(path)
{
#if SLANG_WINDOWS_FAMILY
    // Disable for now on non windows has some problems on *some* Linux based CI.
    {
        String path;
        SlangResult res = Path::getCanonical("source/slang", path);
        SLANG_CHECK(SLANG_SUCCEEDED(res));

        String parentPath;
        res = Path::getCanonical("source", parentPath);
        SLANG_CHECK(SLANG_SUCCEEDED(res));

        String parentPath2 = Path::getParentDirectory(path);
        SLANG_CHECK(parentPath == parentPath2);
    }
#endif
    // Test the paths
    {
        SLANG_CHECK(Path::simplify(".") == ".");
        SLANG_CHECK(Path::simplify("..") == "..");
        SLANG_CHECK(Path::simplify("blah/..") == ".");

        SLANG_CHECK(Path::simplify("blah/.././a") == "a");

        SLANG_CHECK(Path::simplify("a:/what/.././../is/./../this/.") == "a:/../this");

        SLANG_CHECK(Path::simplify("a:/what/.././../is/./../this/./") == "a:/../this");

        SLANG_CHECK(Path::simplify("a:\\what\\..\\.\\..\\is\\.\\..\\this\\.\\") == "a:/../this");

        SLANG_CHECK(
            Path::simplify("tests/preprocessor/.\\pragma-once-a.h") ==
            "tests/preprocessor/pragma-once-a.h");


        SLANG_CHECK(Path::hasRelativeElement("."));
        SLANG_CHECK(Path::hasRelativeElement(".."));
        SLANG_CHECK(Path::hasRelativeElement("blah/.."));

        SLANG_CHECK(Path::hasRelativeElement("blah/.././a"));
        SLANG_CHECK(Path::hasRelativeElement("a") == false);
        SLANG_CHECK(Path::hasRelativeElement("blah/a") == false);
        SLANG_CHECK(Path::hasRelativeElement("a:\\blah/a") == false);


        SLANG_CHECK(Path::hasRelativeElement("a:/what/.././../is/./../this/."));

        SLANG_CHECK(Path::hasRelativeElement("a:/what/.././../is/./../this/./"));

        SLANG_CHECK(Path::hasRelativeElement("a:\\what\\..\\.\\..\\is\\.\\..\\this\\.\\"));
    }
}
