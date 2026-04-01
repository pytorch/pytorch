// test-context.h

#ifndef TEST_CONTEXT_H_INCLUDED
#define TEST_CONTEXT_H_INCLUDED

#include "../../source/compiler-core/slang-downstream-compiler-util.h"
#include "../../source/compiler-core/slang-downstream-compiler.h"
#include "../../source/compiler-core/slang-json-rpc-connection.h"
#include "../../source/core/slang-dictionary.h"
#include "../../source/core/slang-platform.h"
#include "../../source/core/slang-render-api-util.h"
#include "../../source/core/slang-std-writers.h"
#include "../../source/core/slang-string-util.h"
#include "../../source/core/slang-test-tool-util.h"
#include "filecheck.h"
#include "options.h"
#include "slang-com-ptr.h"

#include <mutex>

typedef uint32_t PassThroughFlags;
struct PassThroughFlag
{
    enum Enum : PassThroughFlags
    {
        Dxc = 1 << int(SLANG_PASS_THROUGH_DXC),
        Fxc = 1 << int(SLANG_PASS_THROUGH_FXC),
        Glslang = 1 << int(SLANG_PASS_THROUGH_GLSLANG),
        VisualStudio = 1 << int(SLANG_PASS_THROUGH_VISUAL_STUDIO),
        GCC = 1 << int(SLANG_PASS_THROUGH_GCC),
        Clang = 1 << int(SLANG_PASS_THROUGH_CLANG),
        Generic_C_CPP = 1 << int(SLANG_PASS_THROUGH_GENERIC_C_CPP),
        NVRTC = 1 << int(SLANG_PASS_THROUGH_NVRTC),
        LLVM = 1 << int(SLANG_PASS_THROUGH_LLVM),
        Metal = 1 << int(SLANG_PASS_THROUGH_METAL),
        Tint = 1 << int(SLANG_PASS_THROUGH_TINT),
    };
};

/// Structure that describes requirements needs to run - such as rendering APIs or
/// back-end availability
struct TestRequirements
{

    TestRequirements& addUsedRenderApi(Slang::RenderApiType type)
    {
        using namespace Slang;
        if (type != RenderApiType::Unknown)
        {
            usedRenderApiFlags |= RenderApiFlags(1) << int(type);
        }
        return *this;
    }
    TestRequirements& addUsedBackEnd(SlangPassThrough type)
    {
        if (type != SLANG_PASS_THROUGH_NONE)
        {
            usedBackendFlags |= PassThroughFlags(1) << int(type);
        }
        return *this;
    }
    TestRequirements& addUsedBackends(PassThroughFlags flags)
    {
        usedBackendFlags |= flags;
        return *this;
    }
    TestRequirements& addUsedRenderApis(Slang::RenderApiFlags flags)
    {
        usedRenderApiFlags |= flags;
        return *this;
    }
    /// True if has this render api as used
    bool isUsed(Slang::RenderApiType apiType) const
    {
        return (apiType != Slang::RenderApiType::Unknown) &&
               ((usedRenderApiFlags & (Slang::RenderApiFlags(1) << int(apiType))) != 0);
    }

    Slang::RenderApiType explicitRenderApi =
        Slang::RenderApiType::Unknown;            ///< The render api explicitly specified
    PassThroughFlags usedBackendFlags = 0;        ///< Used backends
    Slang::RenderApiFlags usedRenderApiFlags = 0; ///< Used render api flags (some might be implied)
};

struct FileTestInfo : public Slang::RefObject
{
};

class TestContext
{
public:
    typedef Slang::TestToolUtil::InnerMainFunc InnerMainFunc;

    /// Get the slang session
    SlangSession* getSession() const { return m_session; }

    SlangResult init(const char* exePath);

    /// Get the inner main function (from shared library)
    InnerMainFunc getInnerMainFunc(const Slang::String& dirPath, const Slang::String& name);
    /// Set the function for the shared library
    void setInnerMainFunc(const Slang::String& name, InnerMainFunc func);

    void setTestRequirements(TestRequirements* req);

    TestRequirements* getTestRequirements() const;

    /// If true tests aren't being run just the information on testing is being accumulated
    bool isCollectingRequirements() const { return getTestRequirements() != nullptr; }
    /// If set, then tests are executed
    bool isExecuting() const { return getTestRequirements() == nullptr; }

    /// True if a render API filter is enabled
    bool isRenderApiFilterEnabled() const
    {
        return options.enabledApis != Slang::RenderApiFlag::AllOf && options.enabledApis != 0;
    }

    /// True if a test with the requiredFlags can in principal run (it may not be possible if the
    /// API is not available though)
    bool canRunTestWithRenderApiFlags(Slang::RenderApiFlags requiredFlags);

    /// True if can run unit tests
    bool canRunUnitTests() const { return options.apiOnly == false; }

    /// Given a spawn type, return the final spawn type.
    /// In particular we want 'Default' spawn type to vary by the environment (for example running
    /// on test server on CI)
    SpawnType getFinalSpawnType(SpawnType spawnType);

    SpawnType getFinalSpawnType();

    /// Get compiler set
    Slang::DownstreamCompilerSet* getCompilerSet();
    Slang::IDownstreamCompiler* getDefaultCompiler(SlangSourceLanguage sourceLanguage);

    Slang::JSONRPCConnection* getOrCreateJSONRPCConnection();
    void destroyRPCConnection();

    /// Ctor
    TestContext();
    /// Dtor
    ~TestContext();

    Options options;
    TestCategorySet categorySet;

    /// If set then tests are not run, but their requirements are set

    PassThroughFlags availableBackendFlags = 0;
    Slang::RenderApiFlags availableRenderApiFlags = 0;
    bool isAvailableRenderApiFlagsValid = false;

    Slang::RefPtr<Slang::DownstreamCompilerSet> compilerSet;

    Slang::String exeDirectoryPath;
    Slang::String dllDirectoryPath;
    Slang::String exePath;

    /// Timeout time for communication over connection.
    /// NOTE! If the timeout is hit, the connection will be destroyed, and then recreated.
    /// To test it, compile the core module, if it takes too much time, the core module will be
    /// repeatedly compiled and each time fail.
    /// NOTE! This timeout may be altered in the ctor for a specific target, the initializatoin
    /// value is just the default.
    ///
    /// TODO(JS): We could split the core module compilation from other actions, and have timeout
    /// specific for that. To do this we could have a 'compileCoreModule' RPC method.
    ///
    /// Current default is 60 seconds.
    Slang::Int connectionTimeOutInMs = 60 * 1000;

    void setThreadIndex(int index);
    void setMaxTestRunnerThreadCount(int count);

    void setTestReporter(TestReporter* reporter);
    TestReporter* getTestReporter();
    SlangResult createLanguageServerJSONRPCConnection(Slang::RefPtr<Slang::JSONRPCConnection>& out);

    std::mutex mutex;
    Slang::RefPtr<Slang::JSONRPCConnection> m_languageServerConnection;

    bool isRetry;
    std::mutex mutexFailedTests;
    Slang::List<Slang::RefPtr<FileTestInfo>> failedFileTests;
    Slang::List<Slang::String> failedUnitTests;

    Slang::IFileCheck* getFileCheck() { return m_fileCheck; };

protected:
    SlangResult _createJSONRPCConnection(Slang::RefPtr<Slang::JSONRPCConnection>& out);

    SlangResult locateFileCheck();

    struct SharedLibraryTool
    {
        Slang::ComPtr<ISlangSharedLibrary> m_sharedLibrary;
        InnerMainFunc m_func;
    };

    Slang::List<Slang::RefPtr<Slang::JSONRPCConnection>> m_jsonRpcConnections;
    Slang::List<TestReporter*> m_reporters;
    Slang::List<TestRequirements*> m_testRequirements = nullptr;

    Slang::ComPtr<SlangSession> m_session;

    Slang::Dictionary<Slang::String, SharedLibraryTool> m_sharedLibTools;

    Slang::ComPtr<ISlangSharedLibrary> m_fileCheckLibrary;
    Slang::ComPtr<Slang::IFileCheck> m_fileCheck;
};

#endif // TEST_CONTEXT_H_INCLUDED
