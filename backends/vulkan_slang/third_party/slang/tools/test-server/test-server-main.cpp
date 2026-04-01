// test-server.cpp

#include "../../source/compiler-core/slang-json-rpc-connection.h"
#include "../../source/compiler-core/slang-test-server-protocol.h"
#include "../../source/core/slang-io.h"
#include "../../source/core/slang-process-util.h"
#include "../../source/core/slang-secure-crt.h"
#include "../../source/core/slang-shared-library.h"
#include "../../source/core/slang-string-util.h"
#include "../../source/core/slang-string.h"
#include "../../source/core/slang-test-tool-util.h"
#include "../../source/core/slang-writer.h"
#include "slang-com-helper.h"
#include "test-server-diagnostics.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
// https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/#2.-set-agility-sdk-parameters

extern "C"
{
    __declspec(dllexport) extern const uint32_t D3D12SDKVersion = 711;
}

extern "C"
{
    __declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\";
}
#endif

namespace TestServer
{
using namespace Slang;

class TestReporter : public ITestReporter
{
public:
    // ITestReporter
    virtual SLANG_NO_THROW void SLANG_MCALL startTest(const char* testName) SLANG_OVERRIDE {}
    virtual SLANG_NO_THROW void SLANG_MCALL addResult(TestResult result) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL
    addResultWithLocation(TestResult result, const char* testText, const char* file, int line)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL
    addResultWithLocation(bool testSucceeded, const char* testText, const char* file, int line)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL addExecutionTime(double time) SLANG_OVERRIDE {}
    virtual SLANG_NO_THROW void SLANG_MCALL message(TestMessageType type, const char* message)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL endTest() SLANG_OVERRIDE {}

    StringBuilder m_buf;
    Index m_failCount = 0;
    Index m_testCount = 0;
};

class TestServer
{
public:
    typedef Slang::TestToolUtil::InnerMainFunc InnerMainFunc;

    SlangResult init(int argc, const char* const* argv);

    /// Can return nullptr if cannot create the session
    slang::IGlobalSession* getOrCreateGlobalSession();

    /// Can return nullptr if cannot load the tool
    ISlangSharedLibrary* loadSharedLibrary(const String& name, DiagnosticSink* sink = nullptr);

    /// Get a unit test module. Returns nullptr if not found.
    IUnitTestModule* getUnitTestModule(const String& name, DiagnosticSink* sink = nullptr);

    /// Given a tool name return it's function pointer. Or nullptr on failure.
    InnerMainFunc getToolFunction(const String& name, DiagnosticSink* sink = nullptr);

    /// Execute the server
    SlangResult execute();

    /// Dtor
    ~TestServer();

protected:
    SlangResult _executeSingle();
    SlangResult _executeUnitTest(const JSONRPCCall& call);
    SlangResult _executeTool(const JSONRPCCall& root);

    bool m_quit = false;

    ComPtr<slang::IGlobalSession> m_session; /// The slang session. Is created on demand

    Dictionary<String, ComPtr<ISlangSharedLibrary>>
        m_sharedLibraryMap;                                 ///< Maps tool names to the dll
    Dictionary<String, IUnitTestModule*> m_unitTestModules; ///< All the unit test modules.

    String m_exePath;      ///< Path to executable (including exe name)
    String m_exeDirectory; ///< The directory that holds the exe

    RefPtr<JSONRPCConnection> m_connection; ///< RPC connection, recieves calls to execute and
                                            ///< returns results via JSON-RPC
};

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!! TestServer !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

namespace SlangCTool
{

static void _diagnosticCallback(char const* message, void* userData)
{
    ISlangWriter* writer = (ISlangWriter*)userData;
    writer->write(message, strlen(message));
}

SlangResult innerMain(
    StdWriters* stdWriters,
    slang::IGlobalSession* sharedSession,
    int argc,
    const char* const* argv)
{
    // Assume we will used the shared session
    ComPtr<slang::IGlobalSession> session(sharedSession);

    // The sharedSession always has a pre-loaded core module.
    // This differed test checks if the command line has an option to setup the core module.
    // If so we *don't* use the sharedSession, and create a new session without the core module just
    // for this compilation.
    if (TestToolUtil::hasDeferredCoreModule(Index(argc - 1), argv + 1))
    {
        SLANG_RETURN_ON_FAIL(
            slang_createGlobalSessionWithoutCoreModule(SLANG_API_VERSION, session.writeRef()));
    }

    ComPtr<slang::ICompileRequest> compileRequest;
    SLANG_ALLOW_DEPRECATED_BEGIN
    SLANG_RETURN_ON_FAIL(session->createCompileRequest(compileRequest.writeRef()));
    SLANG_ALLOW_DEPRECATED_END

    // Do any app specific configuration
    for (int i = 0; i < int{SLANG_WRITER_CHANNEL_COUNT_OF}; ++i)
    {
        const auto channel = SlangWriterChannel(i);
        compileRequest->setWriter(channel, stdWriters->getWriter(channel));
    }

    compileRequest->setDiagnosticCallback(
        &_diagnosticCallback,
        stdWriters->getWriter(SLANG_WRITER_CHANNEL_STD_ERROR));
    compileRequest->setCommandLineCompilerMode();

    {
        const SlangResult res = compileRequest->processCommandLineArguments(&argv[1], argc - 1);
        if (SLANG_FAILED(res))
        {
            // TODO: print usage message
            return res;
        }
    }

    SlangResult compileRes = SLANG_OK;

#ifndef _DEBUG
    try
#endif
    {
        // Run the compiler (this will produce any diagnostics through
        // SLANG_WRITER_TARGET_TYPE_DIAGNOSTIC).
        compileRes = compileRequest->compile();

        // If the compilation failed, then get out of here...
        // Turn into an internal Result -> such that return code can be used to vary result to match
        // previous behavior
        compileRes = SLANG_FAILED(compileRes) ? SLANG_E_INTERNAL_FAIL : compileRes;
    }
#ifndef _DEBUG
    catch (const Exception& e)
    {
        WriterHelper writerHelper(stdWriters->getWriter(SLANG_WRITER_CHANNEL_STD_OUTPUT));
        writerHelper.print("internal compiler error: %S\n", e.Message.toWString().begin());
        compileRes = SLANG_FAIL;
    }
#endif

    return compileRes;
}

} // namespace SlangCTool

// SlangITool
#include "../slang-test/slangi-tool-impl.h"

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!! TestServer !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

SlangResult TestServer::init(int argc, const char* const* argv)
{
    m_exePath = argv[0];

    String canonicalPath;
    if (SLANG_SUCCEEDED(Path::getCanonical(m_exePath, canonicalPath)))
    {
        m_exeDirectory = Path::getParentDirectory(canonicalPath);
    }
    else
    {
        m_exeDirectory = Path::getParentDirectory(m_exePath);
    }

    m_connection = new JSONRPCConnection;
    SLANG_RETURN_ON_FAIL(m_connection->initWithStdStreams());
    return SLANG_OK;
}

TestServer::~TestServer()
{
    for (auto& [_, value] : m_unitTestModules)
        value->destroy();
}

slang::IGlobalSession* TestServer::getOrCreateGlobalSession()
{
    if (!m_session)
    {
        // Just create the global session in the regular way if there isn't one set
        SlangGlobalSessionDesc desc = {};
        desc.enableGLSL = true;
        if (SLANG_FAILED(slang_createGlobalSession2(&desc, m_session.writeRef())))
        {
            return nullptr;
        }
        TestToolUtil::setSessionDefaultPreludeFromExePath(m_exePath.getBuffer(), m_session);
    }

    return m_session;
}

ISlangSharedLibrary* TestServer::loadSharedLibrary(const String& name, DiagnosticSink* sink)
{
    ComPtr<ISlangSharedLibrary> lib;
    if (m_sharedLibraryMap.tryGetValue(name, lib))
    {
        return lib;
    }

    auto loader = DefaultSharedLibraryLoader::getSingleton();

    ComPtr<ISlangSharedLibrary> sharedLibrary;
    if (SLANG_FAILED(loader->loadSharedLibrary(name.getBuffer(), sharedLibrary.writeRef())))
    {
        if (sink)
        {
            sink->diagnose(SourceLoc(), ServerDiagnostics::unableToLoadSharedLibrary, name);
        }

        return nullptr;
    }

    m_sharedLibraryMap.add(name, sharedLibrary);
    return sharedLibrary;
}

IUnitTestModule* TestServer::getUnitTestModule(const String& name, DiagnosticSink* sink)
{
    auto unitTestModulePtr = m_unitTestModules.tryGetValue(name);
    if (unitTestModulePtr)
    {
        return *unitTestModulePtr;
    }

    ISlangSharedLibrary* sharedLibrary = loadSharedLibrary(name, sink);
    if (!sharedLibrary)
    {
        return nullptr;
    }

    const char funcName[] = "slangUnitTestGetModule";

    // get the unit test export name
    UnitTestGetModuleFunc getModuleFunc =
        (UnitTestGetModuleFunc)sharedLibrary->findFuncByName(funcName);
    if (!getModuleFunc)
    {
        if (sink)
        {
            sink->diagnose(
                SourceLoc(),
                ServerDiagnostics::unableToFindFunctionInSharedLibrary,
                funcName);
        }
        return nullptr;
    }

    IUnitTestModule* testModule = getModuleFunc();
    if (!testModule)
    {
        if (sink)
        {
            sink->diagnose(SourceLoc(), ServerDiagnostics::unableToGetUnitTestModule);
        }
        return nullptr;
    }

    m_unitTestModules.add(name, testModule);
    return testModule;
}

TestServer::InnerMainFunc TestServer::getToolFunction(const String& name, DiagnosticSink* sink)
{
    if (name == "slangc")
    {
        return &SlangCTool::innerMain;
    }
    else if (name == "slangi")
    {
        return &SlangITool::innerMain;
    }

    StringBuilder sharedLibToolBuilder;
    sharedLibToolBuilder.append(name);
    sharedLibToolBuilder.append("-tool");

    ISlangSharedLibrary* sharedLibrary = loadSharedLibrary(sharedLibToolBuilder, sink);
    if (!sharedLibrary)
    {
        return nullptr;
    }

    const char funcName[] = "innerMain";

    auto func = (InnerMainFunc)sharedLibrary->findFuncByName(funcName);
    if (!func && sink)
    {
        sink->diagnose(
            SourceLoc(),
            ServerDiagnostics::unableToFindFunctionInSharedLibrary,
            funcName);
    }

    return func;
}

SlangResult TestServer::_executeSingle()
{
    // Block waiting for content (or error/closed)
    SLANG_RETURN_ON_FAIL(m_connection->waitForResult());

    // If we don't have a message, we can quit for now
    if (!m_connection->hasMessage())
    {
        return SLANG_OK;
    }

    const JSONRPCMessageType msgType = m_connection->getMessageType();

    switch (msgType)
    {
    case JSONRPCMessageType::Call:
        {
            JSONRPCCall call;
            SLANG_RETURN_ON_FAIL(m_connection->getRPCOrSendError(&call));

            // Do different things
            if (call.method == TestServerProtocol::QuitArgs::g_methodName)
            {
                m_quit = true;
                return SLANG_OK;
            }
            else if (call.method == TestServerProtocol::ExecuteUnitTestArgs::g_methodName)
            {
                SLANG_RETURN_ON_FAIL(_executeUnitTest(call));
                return SLANG_OK;
            }
            else if (call.method == TestServerProtocol::ExecuteToolTestArgs::g_methodName)
            {
                SLANG_RETURN_ON_FAIL(_executeTool(call));
                break;
            }
            else
            {
                return m_connection->sendError(JSONRPC::ErrorCode::MethodNotFound, call.id);
            }
        }
    default:
        {
            return m_connection->sendError(
                JSONRPC::ErrorCode::InvalidRequest,
                m_connection->getCurrentMessageId());
        }
    }

    return SLANG_OK;
}

static Index _findTestIndex(IUnitTestModule* testModule, const String& name)
{
    const auto testCount = testModule->getTestCount();
    for (SlangInt i = 0; i < testCount; ++i)
    {
        auto testName = testModule->getTestName(i);

        if (name == testName)
        {
            return Index(i);
        }
    }
    return -1;
}

SlangResult TestServer::_executeUnitTest(const JSONRPCCall& call)
{
    auto id = m_connection->getPersistentValue(call.id);

    TestServerProtocol::ExecuteUnitTestArgs args;
    SLANG_RETURN_ON_FAIL(m_connection->toNativeArgsOrSendError(call.params, &args, call.id));

    auto sink = m_connection->getSink();

    IUnitTestModule* testModule = getUnitTestModule(args.moduleName, m_connection->getSink());
    if (!testModule)
    {
        sink->diagnose(SourceLoc(), ServerDiagnostics::unableToFindUnitTestModule, args.moduleName);
        return m_connection->sendError(JSONRPC::ErrorCode::InvalidParams, id);
    }

    const Index testIndex = _findTestIndex(testModule, args.testName);
    if (testIndex < 0)
    {
        sink->diagnose(SourceLoc(), ServerDiagnostics::unableToFindTest, args.testName);
        return m_connection->sendError(JSONRPC::ErrorCode::InvalidParams, id);
    }

    TestReporter testReporter;

    testModule->setTestReporter(&testReporter);

    // Assume we will used the shared session
    slang::IGlobalSession* session = getOrCreateGlobalSession();
    if (!session)
    {
        return SLANG_FAIL;
    }

    UnitTestContext unitTestContext;
    unitTestContext.slangGlobalSession = session;
    unitTestContext.workDirectory = "";
    unitTestContext.enabledApis = RenderApiFlags(args.enabledApis);
    unitTestContext.executableDirectory = m_exeDirectory.getBuffer();

    auto testCount = testModule->getTestCount();
    SLANG_ASSERT(testIndex >= 0 && testIndex < testCount);

    UnitTestFunc testFunc = testModule->getTestFunc(testIndex);

    try
    {
        testFunc(&unitTestContext);
    }
    catch (...)
    {
        testReporter.m_failCount++;
    }

    TestServerProtocol::ExecutionResult result;
    result.result = SLANG_OK;

    if (testReporter.m_failCount > 0)
    {
        result.result = SLANG_FAIL;
        result.stdError = testReporter.m_buf.getUnownedSlice();
    }
    else if (testReporter.m_testCount == 0)
    {
        result.result = SLANG_E_NOT_AVAILABLE;
    }

    result.returnCode = int32_t(TestToolUtil::getReturnCode(result.result));
    return m_connection->sendResult(&result, id);
}

SlangResult TestServer::_executeTool(const JSONRPCCall& call)
{
    auto id = m_connection->getPersistentValue(call.id);

    TestServerProtocol::ExecuteToolTestArgs args;

    SLANG_RETURN_ON_FAIL(m_connection->toNativeArgsOrSendError(call.params, &args, id));

    auto sink = m_connection->getSink();

    auto func = getToolFunction(args.toolName, sink);
    if (!func)
    {
        return m_connection->sendError(JSONRPC::ErrorCode::InvalidParams, id);
    }

    // Assume we will used the shared session
    slang::IGlobalSession* session = getOrCreateGlobalSession();
    if (!session)
    {
        return SLANG_FAIL;
    }

    // Work out the args sent to the shared library
    List<const char*> toolArgs;

    // Add the 'exe' name
    toolArgs.add(args.toolName.getBuffer());

    // Add the args
    for (const auto& arg : args.args)
    {
        toolArgs.add(arg.getBuffer());
    }

    StdWriters stdWriters;
    StringBuilder stdOut;
    StringBuilder stdError;

    // Make writer/s act as if they are the console.
    RefPtr<StringWriter> stdOutWriter(new StringWriter(&stdOut, WriterFlag::IsConsole));
    RefPtr<StringWriter> stdErrorWriter(new StringWriter(&stdError, WriterFlag::IsConsole));

    stdWriters.setWriter(SLANG_WRITER_CHANNEL_STD_ERROR, stdErrorWriter);
    stdWriters.setWriter(SLANG_WRITER_CHANNEL_STD_OUTPUT, stdOutWriter);

    // HACK, to make behavior the same as previously
    if (args.toolName == "slangc")
    {
        stdWriters.setWriter(SLANG_WRITER_CHANNEL_DIAGNOSTIC, stdErrorWriter);
    }

    const SlangResult funcRes =
        func(&stdWriters, session, int(toolArgs.getCount()), toolArgs.begin());

    TestServerProtocol::ExecutionResult result;
    result.result = funcRes;
    result.stdError = stdError;
    result.stdOut = stdOut;

    result.returnCode = int32_t(TestToolUtil::getReturnCode(result.result));
    return m_connection->sendResult(&result, id);
}

SlangResult TestServer::execute()
{
    while (m_connection->isActive() && !m_quit)
    {
        // Failure doesn't make the execution terminate
        [[maybe_unused]] const SlangResult res = _executeSingle();
    }

    return SLANG_OK;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TestReporter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

void TestReporter::message(TestMessageType type, const char* message)
{
    if (type == TestMessageType::RunError || type == TestMessageType::TestFailure)
    {
        m_failCount++;
    }

    m_buf << message << "\n";
}

void TestReporter::addResultWithLocation(
    TestResult result,
    const char* testText,
    const char* file,
    int line)
{
    if (result == TestResult::Fail)
    {
        addResultWithLocation(false, testText, file, line);
    }
    else
    {
        m_testCount++;
    }
}

void TestReporter::addResultWithLocation(
    bool testSucceeded,
    const char* testText,
    const char* file,
    int line)
{
    m_testCount++;

    if (testSucceeded)
    {
        return;
    }

    m_buf << "[Failed]: " << testText << "\n";
    m_buf << file << ":" << line << "\n";

    m_failCount++;
}

void TestReporter::addResult(TestResult result)
{
    if (result == TestResult::Fail)
    {
        m_failCount++;
    }
}


SlangResult _execute(int argc, const char* const* argv)
{
    TestServer server;
    SLANG_RETURN_ON_FAIL(server.init(argc, argv));
    SLANG_RETURN_ON_FAIL(server.execute());
    slang::shutdown();
    return SLANG_OK;
}

} // namespace TestServer

int main(int argc, const char* const* argv)
{
    return (int)Slang::TestToolUtil::getReturnCode(TestServer::_execute(argc, argv));
}
