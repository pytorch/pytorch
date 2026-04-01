#include "slang-test-server-protocol.h"

namespace TestServerProtocol
{

static const StructRttiInfo _makeExecuteUnitTestArgsRtti()
{
    ExecuteUnitTestArgs obj;
    StructRttiBuilder builder(&obj, "TestServerProtocol::ExecuteUnitTestArgs", nullptr);

    builder.addField("moduleName", &obj.moduleName);
    builder.addField("testName", &obj.testName);
    builder.addField("enabledApis", &obj.enabledApis);
    return builder.make();
}
/* static */ const UnownedStringSlice ExecuteUnitTestArgs::g_methodName =
    UnownedStringSlice::fromLiteral("unitTest");
/* static */ const StructRttiInfo ExecuteUnitTestArgs::g_rttiInfo = _makeExecuteUnitTestArgsRtti();

static const StructRttiInfo _makeExecuteToolTestArgsRtti()
{
    ExecuteToolTestArgs obj;
    StructRttiBuilder builder(&obj, "TestServerProtocol::ExecuteToolTestArgs", nullptr);
    builder.addField("toolName", &obj.toolName);
    builder.addField("args", &obj.args);
    return builder.make();
}
/* static */ const StructRttiInfo ExecuteToolTestArgs::g_rttiInfo = _makeExecuteToolTestArgsRtti();
/* static */ const UnownedStringSlice ExecuteToolTestArgs::g_methodName =
    UnownedStringSlice::fromLiteral("tool");

static const StructRttiInfo _makeExecutionResultRtti()
{
    ExecutionResult obj;
    StructRttiBuilder builder(&obj, "TestServerProtocol::ExecutionResult", nullptr);
    builder.addField("stdOut", &obj.stdOut);
    builder.addField("stdError", &obj.stdError);
    builder.addField("result", &obj.result);
    builder.addField("returnCode", &obj.returnCode);
    return builder.make();
}
/* static */ const StructRttiInfo ExecutionResult::g_rttiInfo = _makeExecutionResultRtti();

/* static */ const UnownedStringSlice QuitArgs::g_methodName =
    UnownedStringSlice::fromLiteral("quit");

} // namespace TestServerProtocol
