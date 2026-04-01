#ifndef SLANG_COMPILER_CORE_TEST_PROTOCOL_H
#define SLANG_COMPILER_CORE_TEST_PROTOCOL_H

#include "../core/slang-rtti-info.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang-json-value.h"
#include "slang.h"

namespace TestServerProtocol
{

using namespace Slang;

struct ExecuteUnitTestArgs
{
    String moduleName;
    String testName;
    uint32_t enabledApis;

    static const UnownedStringSlice g_methodName;
    static const StructRttiInfo g_rttiInfo;
};

struct ExecuteToolTestArgs
{
    String toolName;   ///< The name of the tool (will be a shared library typically - like
                       ///< render-test). Doesn't need -tool suffix.
    List<String> args; ///< Arguments passed to the tool during exectution

    static const UnownedStringSlice g_methodName;
    static const StructRttiInfo g_rttiInfo;
};

struct QuitArgs
{
    static const UnownedStringSlice g_methodName;
};

struct ExecutionResult
{
    String stdOut;
    String stdError;
    int32_t result = SLANG_OK;
    int32_t returnCode = 0; ///< As returned if invoked as command line

    static const StructRttiInfo g_rttiInfo;
};

} // namespace TestServerProtocol

#endif // SLANG_COMPILER_CORE_TEST_PROTOCOL_H
