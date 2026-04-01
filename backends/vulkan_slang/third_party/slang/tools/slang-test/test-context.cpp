// test-context.cpp
#include "test-context.h"

#include "../../source/compiler-core/slang-language-server-protocol.h"
#include "../../source/core/slang-io.h"
#include "../../source/core/slang-shared-library.h"
#include "../../source/core/slang-string-util.h"
#include "../../source/core/slang-test-tool-util.h"

#include <stdio.h>
#include <stdlib.h>

using namespace Slang;

thread_local int slangTestThreadIndex = 0;

TestContext::TestContext()
{
    /// if we are testing on arm, debug, we may want to increase the connection timeout
#if (SLANG_PROCESSOR_ARM || SLANG_PROCESSOR_ARM_64) && defined(_DEBUG)
    // 10 mins(!). This seems to be the order of time needed for timeout on a CI ARM test system on
    // debug
    connectionTimeOutInMs = 1000 * 60 * 10;
#endif
}

void TestContext::setThreadIndex(int index)
{
    slangTestThreadIndex = index;
}

void TestContext::setMaxTestRunnerThreadCount(int count)
{
    m_jsonRpcConnections.setCount(count);
    m_testRequirements.setCount(count);
    m_reporters.setCount(count);
    for (auto& reporter : m_reporters)
    {
        reporter = nullptr;
    }
}

void TestContext::setTestRequirements(TestRequirements* req)
{
    m_testRequirements[slangTestThreadIndex] = req;
}

TestRequirements* TestContext::getTestRequirements() const
{
    return m_testRequirements[slangTestThreadIndex];
}

void TestContext::setTestReporter(TestReporter* reporter)
{
    m_reporters[slangTestThreadIndex] = reporter;
}

TestReporter* TestContext::getTestReporter()
{
    return m_reporters[slangTestThreadIndex];
}

SlangResult TestContext::locateFileCheck()
{
    DefaultSharedLibraryLoader* loader = DefaultSharedLibraryLoader::getSingleton();

    SLANG_RETURN_ON_FAIL(loader->loadSharedLibrary("slang-llvm", m_fileCheckLibrary.writeRef()));

    if (!m_fileCheckLibrary)
    {
        return SLANG_FAIL;
    }

    using CreateFileCheckFunc = SlangResult (*)(const SlangUUID&, void**);
    auto fn = reinterpret_cast<CreateFileCheckFunc>(
        m_fileCheckLibrary->findFuncByName("createLLVMFileCheck_V1"));
    if (!fn)
    {
        return SLANG_FAIL;
    }
    return fn(SLANG_IID_PPV_ARGS(m_fileCheck.writeRef()));
}

Result TestContext::init(const char* inExePath)
{
    SlangGlobalSessionDesc desc = {};
    desc.enableGLSL = true;
    SLANG_RETURN_ON_FAIL(slang::createGlobalSession(&desc, m_session.writeRef()));
    exePath = inExePath;
    SLANG_RETURN_ON_FAIL(TestToolUtil::getExeDirectoryPath(inExePath, exeDirectoryPath));
    SLANG_RETURN_ON_FAIL(TestToolUtil::getDllDirectoryPath(inExePath, dllDirectoryPath));

    SLANG_RETURN_ON_FAIL(locateFileCheck());

    return SLANG_OK;
}

TestContext::~TestContext()
{
    if (m_languageServerConnection)
    {
        m_languageServerConnection->sendCall(
            LanguageServerProtocol::ExitParams::methodName,
            JSONValue::makeInt(0));
    }
}

TestContext::InnerMainFunc TestContext::getInnerMainFunc(const String& dirPath, const String& name)
{
    {
        SharedLibraryTool* tool = m_sharedLibTools.tryGetValue(name);
        if (tool)
        {
            return tool->m_func;
        }
    }

    StringBuilder sharedLibToolBuilder;
    sharedLibToolBuilder.append(name);
    sharedLibToolBuilder.append("-tool");

    StringBuilder path;
    SharedLibrary::appendPlatformFileName(sharedLibToolBuilder.getUnownedSlice(), path);

    DefaultSharedLibraryLoader* loader = DefaultSharedLibraryLoader::getSingleton();

    SharedLibraryTool tool = {};

    if (SLANG_SUCCEEDED(
            loader->loadPlatformSharedLibrary(path.begin(), tool.m_sharedLibrary.writeRef())))
    {
        tool.m_func = (InnerMainFunc)tool.m_sharedLibrary->findFuncByName("innerMain");
    }

    m_sharedLibTools.add(name, tool);
    return tool.m_func;
}

void TestContext::setInnerMainFunc(const String& name, InnerMainFunc func)
{
    SharedLibraryTool* tool = m_sharedLibTools.tryGetValue(name);
    if (tool)
    {
        tool->m_sharedLibrary.setNull();
        tool->m_func = func;
    }
    else
    {
        SharedLibraryTool tool = {};
        tool.m_func = func;
        m_sharedLibTools.add(name, tool);
    }
}

DownstreamCompilerSet* TestContext::getCompilerSet()
{
    std::lock_guard<std::mutex> lock(mutex);
    if (!compilerSet)
    {
        compilerSet = new DownstreamCompilerSet;

        DownstreamCompilerLocatorFunc locators[int(SLANG_PASS_THROUGH_COUNT_OF)] = {nullptr};

        DownstreamCompilerUtil::setDefaultLocators(locators);
        for (Index i = 0; i < Index(SLANG_PASS_THROUGH_COUNT_OF); ++i)
        {
            auto locator = locators[i];
            if (locator)
            {
                locator(String(), DefaultSharedLibraryLoader::getSingleton(), compilerSet);
            }
        }

        DownstreamCompilerUtil::updateDefaults(compilerSet);
    }
    return compilerSet;
}

SlangResult TestContext::_createJSONRPCConnection(RefPtr<JSONRPCConnection>& out)
{
    RefPtr<Process> process;

    {
        CommandLine cmdLine;
        cmdLine.setExecutableLocation(ExecutableLocation(exeDirectoryPath, "test-server"));
        SLANG_RETURN_ON_FAIL(Process::create(
            cmdLine,
            Process::Flag::AttachDebugger | Process::Flag::DisableStdErrRedirection,
            process));
    }

    Stream* writeStream = process->getStream(StdStreamType::In);
    RefPtr<BufferedReadStream> readStream(
        new BufferedReadStream(process->getStream(StdStreamType::Out)));
    RefPtr<BufferedReadStream> readErrStream(
        new BufferedReadStream(process->getStream(StdStreamType::ErrorOut)));

    RefPtr<HTTPPacketConnection> connection = new HTTPPacketConnection(readStream, writeStream);
    RefPtr<JSONRPCConnection> rpcConnection = new JSONRPCConnection;

    SLANG_RETURN_ON_FAIL(
        rpcConnection->init(connection, JSONRPCConnection::CallStyle::Default, process));

    out = rpcConnection;

    return SLANG_OK;
}

SlangResult TestContext::createLanguageServerJSONRPCConnection(RefPtr<JSONRPCConnection>& out)
{
    RefPtr<Process> process;

    {
        CommandLine cmdLine;
        cmdLine.setExecutableLocation(ExecutableLocation(exeDirectoryPath, "slangd"));
        SLANG_RETURN_ON_FAIL(Process::create(cmdLine, Process::Flag::AttachDebugger, process));
    }

    Stream* writeStream = process->getStream(StdStreamType::In);
    RefPtr<BufferedReadStream> readStream(
        new BufferedReadStream(process->getStream(StdStreamType::Out)));

    RefPtr<HTTPPacketConnection> connection = new HTTPPacketConnection(readStream, writeStream);
    RefPtr<JSONRPCConnection> rpcConnection = new JSONRPCConnection;

    SLANG_RETURN_ON_FAIL(
        rpcConnection->init(connection, JSONRPCConnection::CallStyle::Object, process));

    out = rpcConnection;

    return SLANG_OK;
}

void TestContext::destroyRPCConnection()
{
    if (m_jsonRpcConnections[slangTestThreadIndex])
    {
        m_jsonRpcConnections[slangTestThreadIndex]->disconnect();
        m_jsonRpcConnections[slangTestThreadIndex].setNull();
    }
}

Slang::JSONRPCConnection* TestContext::getOrCreateJSONRPCConnection()
{
    if (!m_jsonRpcConnections[slangTestThreadIndex])
    {
        if (SLANG_FAILED(_createJSONRPCConnection(m_jsonRpcConnections[slangTestThreadIndex])))
        {
            return nullptr;
        }
    }

    return m_jsonRpcConnections[slangTestThreadIndex];
}


Slang::IDownstreamCompiler* TestContext::getDefaultCompiler(SlangSourceLanguage sourceLanguage)
{
    DownstreamCompilerSet* set = getCompilerSet();
    return set ? set->getDefaultCompiler(sourceLanguage) : nullptr;
}

bool TestContext::canRunTestWithRenderApiFlags(Slang::RenderApiFlags requiredFlags)
{
    // If only allow tests that use API - then the requiredFlags must be 0
    if (options.apiOnly && requiredFlags == 0)
    {
        return false;
    }
    // Are the required rendering APIs enabled from the -api command line switch
    return (requiredFlags & options.enabledApis) == requiredFlags;
}

SpawnType TestContext::getFinalSpawnType(SpawnType spawnType)
{
    if (spawnType == SpawnType::Default)
    {
        if (options.outputMode == TestOutputMode::Default)
        {
            return SpawnType::UseSharedLibrary;
        }
        else
        {
            return SpawnType::UseTestServer;
        }
    }

    // Just return whatever spawnType was passed in
    return spawnType;
}

SpawnType TestContext::getFinalSpawnType()
{
    return getFinalSpawnType(options.defaultSpawnType);
}
