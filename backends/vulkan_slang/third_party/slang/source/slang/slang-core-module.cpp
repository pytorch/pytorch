#include "../core/slang-string-util.h"
#include "slang-compiler.h"
#include "slang-ir.h"

#define STRINGIZE(x) STRINGIZE2(x)
#define STRINGIZE2(x) #x
#define LINE_STRING STRINGIZE(__LINE__)

namespace Slang
{
String Session::getCoreModulePath()
{
    if (coreModulePath.getLength() == 0)
    {
        // Make sure we have a line of text from __FILE__, that we'll extract the filename from
        List<UnownedStringSlice> lines;
        StringUtil::calcLines(UnownedStringSlice::fromLiteral(__FILE__), lines);
        SLANG_ASSERT(lines.getCount() > 0 && lines[0].getLength() > 0);

        // Make the path just the filename to remove issues around path being included on different
        // targets
        coreModulePath = Path::getFileName(lines[0]);
    }
    return coreModulePath;
}
} // namespace Slang
