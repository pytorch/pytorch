// slang-test-main.cpp

#include "../../source/compiler-core/slang-artifact-desc-util.h"
#include "../../source/compiler-core/slang-artifact-helper.h"
#include "../../source/core/slang-byte-encode-util.h"
#include "../../source/core/slang-castable.h"
#include "../../source/core/slang-char-util.h"
#include "../../source/core/slang-hex-dump-util.h"
#include "../../source/core/slang-io.h"
#include "../../source/core/slang-memory-arena.h"
#include "../../source/core/slang-process-util.h"
#include "../../source/core/slang-render-api-util.h"
#include "../../source/core/slang-shared-library.h"
#include "../../source/core/slang-std-writers.h"
#include "../../source/core/slang-string-escape-util.h"
#include "../../source/core/slang-string-util.h"
#include "../../source/core/slang-token-reader.h"
#include "../../source/core/slang-type-text-util.h"
#include "slang-com-helper.h"
#include "unit-test/slang-unit-test.h"
#undef SLANG_UNIT_TEST

#include "../../source/compiler-core/slang-artifact-associated-impl.h"
#include "../../source/compiler-core/slang-downstream-compiler.h"
#include "../../source/compiler-core/slang-language-server-protocol.h"
#include "../../source/compiler-core/slang-nvrtc-compiler.h"
#include "directory-util.h"
#include "options.h"
#include "parse-diagnostic-util.h"
#include "slangc-tool.h"
#include "slangi-tool.h"
#include "test-context.h"
#include "test-reporter.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#define SLANG_PRELUDE_NAMESPACE CPPPrelude
#include "../../prelude/slang-cpp-types.h"

#include <atomic>
#include <thread>

using namespace Slang;

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

// Options for a particular test
struct TestOptions
{
    enum Type
    {
        Normal,     ///< A regular test
        Diagnostic, ///< Diagnostic tests will always run (as form of failure is being tested)
    };

    void addCategory(TestCategory* category)
    {
        if (categories.indexOf(category) < 0)
        {
            categories.add(category);
        }
    }
    void addCategories(TestCategory* const* inCategories, Index count)
    {
        for (Index i = 0; i < count; ++i)
        {
            addCategory(inCategories[i]);
        }
    }

    // Small helper to help consistently interrogating for filecheck usage
    bool getFileCheckPrefix(String& prefix) const
    {
        return commandOptions.tryGetValue("filecheck", prefix);
    }
    bool getFileCheckBufferPrefix(String& prefix) const
    {
        return commandOptions.tryGetValue("filecheck-buffer", prefix);
    }

    Type type = Type::Normal;

    String command;
    List<String> args;

    Dictionary<String, String> commandOptions;

    // The categories that this test was assigned to
    List<TestCategory*> categories;

    bool isEnabled = true;
    bool isSynthesized = false;
};

struct FileTestInfoImpl : public FileTestInfo
{
    String testName;
    String filePath;
    String outputStem;
    TestOptions options;
};

struct TestDetails
{
    TestDetails() {}
    explicit TestDetails(const TestOptions& inOptions)
        : options(inOptions)
    {
    }

    TestOptions options;           ///< The options for the test
    TestRequirements requirements; ///< The requirements for the test to work
};

// Information on tests to run for a particular file
struct FileTestList
{
    List<TestDetails> tests;
};


struct TestInput
{
    // Path to the input file for the test
    String filePath;

    // Prefix for the path that test output should write to
    // (usually the same as `filePath`, but will differ when
    // we run multiple tests out of the same file)
    String outputStem;

    // Arguments for the test (usually to be interpreted
    // as command line args)
    TestOptions const* testOptions;

    // Determines how the test will be spawned
    SpawnType spawnType;
};

typedef TestResult (*TestCallback)(TestContext* context, TestInput& input);

// Globals

// Pre declare
static void _addRenderTestOptions(const Options& options, CommandLine& ioCmdLine);

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!! Functions !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

// Tries to read in order
// - The file specific to this test:      input.outputStem + suffix
// - The general file for a set of tests: input.fileName + suffix;
static SlangResult _readTestFile(const TestInput& input, const String& suffix, String& out)
{
    StringBuilder buf;
    buf << input.outputStem << suffix;
    if (auto r = Slang::File::readAllText(buf, out); SLANG_SUCCEEDED(r))
    {
        return r;
    }

    buf.clear();
    buf << input.filePath << suffix;
    return Slang::File::readAllText(buf, out);
}


bool match(char const** ioCursor, char const* expected)
{
    char const* cursor = *ioCursor;
    while (*expected && *cursor == *expected)
    {
        cursor++;
        expected++;
    }
    if (*expected != 0)
        return false;

    *ioCursor = cursor;
    return true;
}

void skipHorizontalSpace(char const** ioCursor)
{
    char const* cursor = *ioCursor;
    for (;;)
    {
        switch (*cursor)
        {
        case ' ':
        case '\t':
            cursor++;
            continue;

        default:
            break;
        }

        break;
    }
    *ioCursor = cursor;
}

void skipToEndOfLine(char const** ioCursor)
{
    char const* cursor = *ioCursor;
    for (;;)
    {
        int c = *cursor;
        switch (c)
        {
        default:
            cursor++;
            continue;

        case '\r':
        case '\n':
            {
                cursor++;
                int d = *cursor;
                if ((c ^ d) == ('\r' ^ '\n'))
                {
                    cursor++;
                }
            }
            [[fallthrough]];
        case 0:
            *ioCursor = cursor;
            return;
        }
    }
}

String getString(char const* textBegin, char const* textEnd)
{
    StringBuilder sb;
    sb.append(textBegin, textEnd - textBegin);
    return sb.produceString();
}

String collectRestOfLine(char const** ioCursor)
{
    char const* cursor = *ioCursor;

    char const* textBegin = cursor;
    skipToEndOfLine(&cursor);
    char const* textEnd = cursor;

    *ioCursor = cursor;
    return getString(textBegin, textEnd);
}

static bool _isEndOfLineOrParens(char c)
{
    switch (c)
    {
    case '\n':
    case '\r':
    case 0:
    case ')':
        {
            return true;
        }
    default:
        return false;
    }
}

static SlangResult _parseCategories(
    TestCategorySet* categorySet,
    char const** ioCursor,
    TestOptions& out)
{
    char const* cursor = *ioCursor;

    // If don't have ( we don't have category list
    if (*cursor == '(')
    {
        cursor++;
        const char* const start = cursor;

        // Find the end
        for (; !_isEndOfLineOrParens(*cursor); ++cursor)
            ;
        if (*cursor != ')')
        {
            *ioCursor = cursor;
            return SLANG_FAIL;
        }
        cursor++;

        List<UnownedStringSlice> slices;
        StringUtil::split(UnownedStringSlice(start, cursor - 1), ',', slices);

        for (auto& slice : slices)
        {
            // Trim any whitespace
            auto categoryName = slice.trim();

            TestCategory* category = categorySet->find(categoryName);

            if (!category)
            {
                // Mark this test as disabled, as we don't have all of the categories
                out.isEnabled = false;
                break;
            }

            out.addCategory(category);
        }
    }

    *ioCursor = cursor;
    return SLANG_OK;
}

static SlangResult _parseCommandArguments(char const** ioCursor, TestOptions& out)
{
    char const* cursor = *ioCursor;

    // If don't have ( we don't have any additional options
    if (*cursor == '(')
    {
        cursor++;
        const char* const start = cursor;

        // Find the end
        for (; !_isEndOfLineOrParens(*cursor); ++cursor)
            ;
        if (*cursor != ')')
        {
            *ioCursor = cursor;
            return SLANG_FAIL;
        }
        cursor++;

        List<UnownedStringSlice> options;
        StringUtil::split(UnownedStringSlice(start, cursor - 1), ',', options);

        for (auto& option : options)
        {
            auto i = option.indexOf('=');
            if (i == -1)
            {
                out.commandOptions.add(option.trim(), "");
            }
            else
            {
                out.commandOptions.add(option.head(i).trim(), option.tail(i + 1).trim());
            }
        }
    }

    *ioCursor = cursor;
    return SLANG_OK;
}

static SlangResult _parseArg(const char** ioCursor, UnownedStringSlice& outArg)
{
    const char* cursor = *ioCursor;
    const char* const argBegin = cursor;

    // Let's try to read one option
    for (;;)
    {
        switch (*cursor)
        {
        default:
            {
                ++cursor;
                break;
            }
        case '"':
            {
                // If we have quotes let's just parse them as is and make output
                auto escapeHandler = StringEscapeUtil::getHandler(StringEscapeUtil::Style::Space);
                SLANG_RETURN_ON_FAIL(escapeHandler->lexQuoted(cursor, &cursor));
                break;
            }
        case 0:
        case '\r':
        case '\n':
        case ' ':
        case '\t':
            {
                char const* argEnd = cursor;
                assert(argBegin != argEnd);

                outArg = UnownedStringSlice(argBegin, argEnd);
                *ioCursor = cursor;
                return SLANG_OK;
            }
        }
    }
}

static SlangResult _gatherTestOptions(
    TestCategorySet* categorySet,
    char const** ioCursor,
    TestOptions& outOptions)
{
    SLANG_RETURN_ON_FAIL(_parseCategories(categorySet, ioCursor, outOptions));

    char const* cursor = *ioCursor;

    if (*cursor != ':')
    {
        return SLANG_FAIL;
    }
    cursor++;

    // Next scan for a sub-command name
    char const* commandStart = cursor;
    for (;;)
    {
        switch (*cursor)
        {
        default:
            cursor++;
            continue;

        case '(':
        case ':':
            break;

        case 0:
        case '\r':
        case '\n':
            return SLANG_FAIL;
        }

        break;
    }
    char const* commandEnd = cursor;

    outOptions.command = getString(commandStart, commandEnd);

    // Allow parameterizing the test command separately from the arguments, this
    // is because the arguments are often passed to the compiler verbatim, and
    // it's messy to have the test runner rifling through and picking things
    // out
    // Format is: (foo=bar, baz = 2)
    SLANG_RETURN_ON_FAIL(_parseCommandArguments(&cursor, outOptions));

    if (*cursor == ':')
        cursor++;
    else
    {
        return SLANG_FAIL;
    }

    // Now scan for arguments. For now we just assume that
    // any whitespace separation indicates a new argument
    for (;;)
    {
        skipHorizontalSpace(&cursor);

        // End of line? then no more options.
        switch (*cursor)
        {
        case 0:
        case '\r':
        case '\n':
            skipToEndOfLine(&cursor);

            *ioCursor = cursor;
            return SLANG_OK;

        default:
            break;
        }

        // Let's try to read one option
        UnownedStringSlice arg;
        SLANG_RETURN_ON_FAIL(_parseArg(&cursor, arg));

        outOptions.args.add(arg);
    }
}


static RenderApiFlags _getRequiredRenderApisByCommand(const UnownedStringSlice& name);

static void _combineOptions(
    TestCategorySet* categorySet,
    const TestOptions& fileOptions,
    TestOptions& ioOptions)
{
    // And the file categories
    ioOptions.addCategories(fileOptions.categories.getBuffer(), fileOptions.categories.getCount());

    // If no categories were specified, then add the default category
    if (ioOptions.categories.getCount() == 0)
    {
        ioOptions.categories.add(categorySet->defaultCategory);
    }
}

static SlangResult _extractCommand(const char** ioCursor, UnownedStringSlice& outCommand)
{
    const char* cursor = *ioCursor;
    const char* const start = cursor;

    while (true)
    {
        const char c = *cursor;

        if (CharUtil::isAlpha(c) || c == '_')
        {
            cursor++;
            continue;
        }

        if (c == ':' || c == '(' || c == 0 || c == '\n' || c == '\r')
        {
            *ioCursor = cursor;
            outCommand = UnownedStringSlice(start, cursor);
            return SLANG_OK;
        }

        return SLANG_FAIL;
    }
}

static void applyMacroSubstitution(String filePath, TestDetails& details)
{
    for (auto& arg : details.options.args)
    {
        arg = StringUtil::replaceAll(
            arg.getUnownedSlice(),
            toSlice("$dirname"),
            Path::getParentDirectory(filePath).getUnownedSlice());
    }
}

// Try to read command-line options from the test file itself
static SlangResult _gatherTestsForFile(
    TestCategorySet* categorySet,
    String filePath,
    FileTestList* outTestList)
{
    outTestList->tests.clear();

    String fileContents;

    SLANG_RETURN_ON_FAIL(Slang::File::readAllText(filePath, fileContents));

    // Walk through the lines of the file, looking for test commands
    char const* cursor = fileContents.begin();

    // Options that are specified across all tests in the file.
    TestOptions fileOptions;

    while (*cursor)
    {
        // We are at the start of a line of input.

        skipHorizontalSpace(&cursor);

        if (!match(&cursor, "//"))
        {
            skipToEndOfLine(&cursor);
            continue;
        }

        UnownedStringSlice command;

        if (SLANG_FAILED(_extractCommand(&cursor, command)))
        {
            // Couldn't find a command so skip
            skipToEndOfLine(&cursor);
            continue;
        }

        // Look for a pattern that matches what we want
        if (command == "TEST_IGNORE_FILE")
        {
            outTestList->tests.clear();
            return SLANG_OK;
        }

        const UnownedStringSlice disablePrefix = UnownedStringSlice::fromLiteral("DISABLE_");

        TestDetails testDetails;

        {
            if (command.startsWith(disablePrefix))
            {
                testDetails.options.isEnabled = false;
                command = command.tail(disablePrefix.getLength());
            }
        }

        if (command == "TEST_CATEGORY")
        {
            SlangResult res = _parseCategories(categorySet, &cursor, fileOptions);

            // If if failed we are done, unless it was just 'not available'
            if (SLANG_FAILED(res) && res != SLANG_E_NOT_AVAILABLE)
                return res;

            skipToEndOfLine(&cursor);
            continue;
        }

        if (command == "TEST")
        {
            SLANG_RETURN_ON_FAIL(_gatherTestOptions(categorySet, &cursor, testDetails.options));
            applyMacroSubstitution(filePath, testDetails);

            // See if the type of test needs certain APIs available
            const RenderApiFlags testRequiredApis =
                _getRequiredRenderApisByCommand(testDetails.options.command.getUnownedSlice());
            testDetails.requirements.addUsedRenderApis(testRequiredApis);

            // Apply the file wide options
            _combineOptions(categorySet, fileOptions, testDetails.options);

            outTestList->tests.add(testDetails);
        }
        else if (command == "DIAGNOSTIC_TEST")
        {
            SLANG_RETURN_ON_FAIL(_gatherTestOptions(categorySet, &cursor, testDetails.options));
            applyMacroSubstitution(filePath, testDetails);

            // Apply the file wide options
            _combineOptions(categorySet, fileOptions, testDetails.options);

            // Mark that it is a diagnostic test
            testDetails.options.type = TestOptions::Type::Diagnostic;
            outTestList->tests.add(testDetails);
        }
        else
        {
            // Hmm we don't know what kind of test this actually is.
            // Assume that's ok and this *isn't* a test and ignore.
            skipToEndOfLine(&cursor);
        }
    }

    return SLANG_OK;
}

static void SLANG_STDCALL _fileCheckDiagnosticCallback(
    void* data,
    const TestMessageType messageType,
    const char* message) noexcept
{
    auto& testReporter = *reinterpret_cast<TestReporter*>(data);
    testReporter.message(messageType, message);
}

//
// Check some generated output with FileCheck
//
static TestResult _fileCheckTest(
    TestContext& context,
    const String& fileCheckRules,
    const String& fileCheckPrefix,
    const String& outputToCheck)
{
    auto& testReporter = *context.getTestReporter();

    IFileCheck* fc = context.getFileCheck();
    if (!fc)
    {
        // Ignore if FileCheck is not available.
        // We could report an error, but our ARM64 CI doesn't have FileCheck yet.
        testReporter.message(TestMessageType::Info, "FileCheck is not available");
        return TestResult::Ignored;
    }

    const bool coloredOutput = true;
    testReporter.message(TestMessageType::Info, outputToCheck.getBuffer());
    return fc->performTest(
        "slang-test",
        fileCheckRules.begin(),
        fileCheckPrefix.begin(),
        outputToCheck.begin(),
        "actual-output",
        _fileCheckDiagnosticCallback,
        &testReporter,
        coloredOutput);
}

template<typename Compare>
static TestResult _fileComparisonTest(
    TestContext& context,
    const TestInput& input,
    const char* defaultExpectedContent,
    const char* expectedFileSuffix,
    const String& actualOutput,
    Compare compare)
{
    String expectedOutput;

    if (SLANG_FAILED(_readTestFile(input, expectedFileSuffix, expectedOutput)))
    {
        if (defaultExpectedContent)
        {
            expectedOutput = defaultExpectedContent;
        }
        else
        {
            context.getTestReporter()->messageFormat(
                TestMessageType::RunError,
                "Unable to read %s output for '%s'\n",
                expectedFileSuffix,
                input.outputStem.getBuffer());
            return TestResult::Fail;
        }
    }

    // Otherwise we compare to the expected output
    if (!compare(actualOutput, expectedOutput))
    {
        context.getTestReporter()->dumpOutputDifference(expectedOutput, actualOutput);
        return TestResult::Fail;
    }
    return TestResult::Pass;
}

static bool _areLinesEqual(const String& a, const String& e)
{
    return StringUtil::areLinesEqual(a.getUnownedSlice(), e.getUnownedSlice());
}

// Either run FileCheck over the result, or read and compare with a .expected file
// On a comparison failure, dump the difference
// On any failure, write a .actual file.
template<typename Compare = decltype(_areLinesEqual)>
static TestResult _validateOutput(
    TestContext* const context,
    const TestInput& input,
    const String& actualOutput,
    const bool forceFailure = false,
    const char* defaultExpectedContent = nullptr,
    const Compare compare = _areLinesEqual)
{
    String fileCheckPrefix;
    const TestResult result =
        input.testOptions->getFileCheckPrefix(fileCheckPrefix)
            ? _fileCheckTest(*context, input.filePath, fileCheckPrefix, actualOutput)
            : _fileComparisonTest(
                  *context,
                  input,
                  defaultExpectedContent,
                  ".expected",
                  actualOutput,
                  compare);

    // If the test failed, then we write the actual output to a file
    // so that we can easily diff it from the command line and
    // diagnose the problem.
    if (result == TestResult::Fail || forceFailure)
    {
        String actualOutputPath = input.outputStem + ".actual";
        Slang::File::writeAllText(actualOutputPath, actualOutput);
        return TestResult::Fail;
    }
    else
    {
        return result;
    }
}

Result spawnAndWaitExe(
    TestContext* context,
    const String& testPath,
    const CommandLine& cmdLine,
    ExecuteResult& outRes)
{
    std::lock_guard<std::mutex> lock(context->mutex);

    const auto& options = context->options;

    if (options.shouldBeVerbose)
    {
        String commandLine = cmdLine.toString();
        context->getTestReporter()->messageFormat(
            TestMessageType::Info,
            "%s\n",
            commandLine.begin());
    }

    Result res = ProcessUtil::execute(cmdLine, outRes);
    if (SLANG_FAILED(res))
    {
        //        fprintf(stderr, "failed to run test '%S'\n", testPath.ToWString());
        context->getTestReporter()->messageFormat(
            TestMessageType::RunError,
            "failed to run test '%S'",
            testPath.toWString().begin());
    }
    return res;
}


Result spawnAndWaitSharedLibrary(
    TestContext* context,
    const String& testPath,
    const CommandLine& cmdLine,
    ExecuteResult& outRes)
{
    std::lock_guard<std::mutex> lock(context->mutex);

    const auto& options = context->options;
    String exeName = Path::getFileNameWithoutExt(cmdLine.m_executableLocation.m_pathOrName);

    if (options.shouldBeVerbose)
    {
        CommandLine testCmdLine;

        testCmdLine.setExecutableLocation(ExecutableLocation("slang-test"));

        if (options.binDir.getLength())
        {
            testCmdLine.addArg("-bindir");
            testCmdLine.addArg(options.binDir);
        }

        testCmdLine.addArg(exeName);
        testCmdLine.m_args.addRange(cmdLine.m_args);

        context->getTestReporter()->messageFormat(
            TestMessageType::Info,
            "%s\n",
            testCmdLine.toString().getBuffer());
    }

    auto func = context->getInnerMainFunc(context->options.binDir, exeName);
    if (func)
    {
        StringBuilder stdErrorString;
        StringBuilder stdOutString;

        // Say static so not released
        StringWriter stdError(&stdErrorString, WriterFlag::IsConsole | WriterFlag::IsStatic);
        StringWriter stdOut(&stdOutString, WriterFlag::IsConsole | WriterFlag::IsStatic);

        StdWriters* prevStdWriters = StdWriters::getSingleton();

        StdWriters stdWriters;
        stdWriters.setWriter(SLANG_WRITER_CHANNEL_STD_ERROR, &stdError);
        stdWriters.setWriter(SLANG_WRITER_CHANNEL_STD_OUTPUT, &stdOut);

        if (exeName == "slangc" || exeName == "slangi")
        {
            stdWriters.setWriter(SLANG_WRITER_CHANNEL_DIAGNOSTIC, &stdError);
        }

        String exePath = Path::combine(context->exeDirectoryPath, exeName);

        List<const char*> args;
        args.add(exePath.getBuffer());
        for (const auto& cmdArg : cmdLine.m_args)
        {
            args.add(cmdArg.getBuffer());
        }

        SlangResult res =
            func(&stdWriters, context->getSession(), int(args.getCount()), args.begin());

        StdWriters::setSingleton(prevStdWriters);

        outRes.standardError = stdErrorString;
        outRes.standardOutput = stdOutString;

        outRes.resultCode = (int)TestToolUtil::getReturnCode(res);

        return SLANG_OK;
    }

    return SLANG_FAIL;
}


Result spawnAndWaitProxy(
    TestContext* context,
    const String& testPath,
    const CommandLine& inCmdLine,
    ExecuteResult& outRes)
{
    std::lock_guard<std::mutex> lock(context->mutex);

    // Get the name of the thing to execute
    String exeName = Path::getFileNameWithoutExt(inCmdLine.m_executableLocation.m_pathOrName);

    if (exeName == "slangc" || exeName == "slangi")
    {
        // If the test is slangc there is a command line version we can just directly use
        // return spawnAndWaitExe(context, testPath, inCmdLine, outRes);
        return spawnAndWaitSharedLibrary(context, testPath, inCmdLine, outRes);
    }

    CommandLine cmdLine(inCmdLine);

    // Make the first arg the name of the tool to invoke
    cmdLine.m_args.insert(0, exeName);
    cmdLine.setExecutableLocation(ExecutableLocation(context->exeDirectoryPath, "test-proxy"));

    const auto& options = context->options;
    if (options.shouldBeVerbose)
    {
        String commandLine = cmdLine.toString();
        context->getTestReporter()->messageFormat(
            TestMessageType::Info,
            "%s\n",
            commandLine.begin());
    }

    // Execute
    Result res = ProcessUtil::execute(cmdLine, outRes);
    if (SLANG_FAILED(res))
    {
        //        fprintf(stderr, "failed to run test '%S'\n", testPath.ToWString());
        context->getTestReporter()->messageFormat(
            TestMessageType::RunError,
            "failed to run test '%S'",
            testPath.toWString().begin());
    }

    return res;
}

static Result _executeRPC(
    TestContext* context,
    SpawnType spawnType,
    const UnownedStringSlice& method,
    const RttiInfo* rttiInfo,
    const void* args,
    ExecuteResult& outRes)
{
    // If we are 'fully isolated', we cannot share a test server.
    // So tear down the RPC connection if there is one currently.
    if (spawnType == SpawnType::UseFullyIsolatedTestServer)
    {
        context->destroyRPCConnection();
    }

    JSONRPCConnection* rpcConnection = context->getOrCreateJSONRPCConnection();
    if (!rpcConnection)
    {
        return SLANG_FAIL;
    }

    // Execute
    if (SLANG_FAILED(rpcConnection->sendCall(method, rttiInfo, args)))
    {
        context->destroyRPCConnection();
        return SLANG_FAIL;
    }

    // Wait for the result
    rpcConnection->waitForResult(context->connectionTimeOutInMs);

    if (!rpcConnection->hasMessage())
    {
        // We can assume somethings gone wrong. So lets kill the connection and fail.
        context->destroyRPCConnection();
        return SLANG_FAIL;
    }

    if (rpcConnection->getMessageType() != JSONRPCMessageType::Result)
    {
        context->destroyRPCConnection();
        return SLANG_FAIL;
    }

    // Get the result
    TestServerProtocol::ExecutionResult exeRes;
    if (SLANG_FAILED(rpcConnection->getMessage(&exeRes)))
    {
        context->destroyRPCConnection();
        return SLANG_FAIL;
    }

    outRes.resultCode = exeRes.returnCode;
    outRes.standardError = exeRes.stdError;
    outRes.standardOutput = exeRes.stdOut;

    return SLANG_OK;
}

template<typename T>
static Result _executeRPC(
    TestContext* context,
    SpawnType spawnType,
    const UnownedStringSlice& method,
    const T* msg,
    ExecuteResult& outRes)
{
    return _executeRPC(context, spawnType, method, GetRttiInfo<T>::get(), (const void*)msg, outRes);
}

Result spawnAndWaitTestServer(
    TestContext* context,
    SpawnType spawnType,
    const String& testPath,
    const CommandLine& inCmdLine,
    ExecuteResult& outRes)
{
    String exeName = Path::getFileNameWithoutExt(inCmdLine.m_executableLocation.m_pathOrName);

    // This is a test tool execution
    TestServerProtocol::ExecuteToolTestArgs args;

    args.toolName = exeName;
    args.args = inCmdLine.m_args;

    return _executeRPC(
        context,
        spawnType,
        TestServerProtocol::ExecuteToolTestArgs::g_methodName,
        &args,
        outRes);
}

static SlangResult _extractArg(const CommandLine& cmdLine, const String& argName, String& outValue)
{
    SLANG_ASSERT(argName.getLength() > 0 && argName[0] == '-');
    Index index = cmdLine.findArgIndex(argName.getUnownedSlice());

    if (index >= 0 && index < cmdLine.getArgCount() - 1)
    {
        outValue = cmdLine.m_args[index + 1];
        return SLANG_OK;
    }
    return SLANG_FAIL;
}

static bool _hasOption(const List<String>& args, const String& argName)
{
    return args.indexOf(argName) != Index(-1);
}

static PassThroughFlags _getPassThroughFlagsForTarget(SlangCompileTarget target)
{
    switch (target)
    {
    case SLANG_TARGET_UNKNOWN:

    case SLANG_HLSL:
    case SLANG_GLSL:
    case SLANG_C_SOURCE:
    case SLANG_CPP_SOURCE:
    case SLANG_CPP_PYTORCH_BINDING:
    case SLANG_HOST_CPP_SOURCE:
    case SLANG_CUDA_SOURCE:
    case SLANG_METAL:
    case SLANG_WGSL:
    case SLANG_HOST_VM:
        {
            return 0;
        }
    case SLANG_WGSL_SPIRV:
    case SLANG_WGSL_SPIRV_ASM:
        {
            return PassThroughFlag::Tint;
        }
    case SLANG_DXBC:
    case SLANG_DXBC_ASM:
        {
            return PassThroughFlag::Fxc;
        }
    case SLANG_SPIRV:
    case SLANG_SPIRV_ASM:
        {
            return PassThroughFlag::Glslang;
        }
    case SLANG_DXIL:
    case SLANG_DXIL_ASM:
        {
            return PassThroughFlag::Dxc;
        }

    case SLANG_METAL_LIB:
    case SLANG_METAL_LIB_ASM:
        {
            return PassThroughFlag::Metal;
        }

    case SLANG_SHADER_HOST_CALLABLE:
    case SLANG_HOST_HOST_CALLABLE:

    case SLANG_HOST_EXECUTABLE:
    case SLANG_SHADER_SHARED_LIBRARY:
    case SLANG_HOST_SHARED_LIBRARY:
        {
            return PassThroughFlag::Generic_C_CPP;
        }
    case SLANG_PTX:
        {
            return PassThroughFlag::NVRTC;
        }

    default:
        {
            SLANG_ASSERT(!"Unknown type");
            return 0;
        }
    }
}

static SlangResult _extractRenderTestRequirements(
    const CommandLine& cmdLine,
    TestRequirements* ioRequirements)
{
    const auto& args = cmdLine.m_args;

    // TODO(JS):
    // This is rather convoluted in that it has to work out from the command line parameters passed
    // to render-test what renderer will be used.
    // That a similar logic has to be kept inside the implementation of render-test and both this
    // and render-test will have to be kept in sync.

    bool useDxil = cmdLine.findArgIndex(UnownedStringSlice::fromLiteral("-use-dxil")) >= 0;

    bool usePassthru = false;

    // Work out what kind of render will be used
    RenderApiType renderApiType;
    {
        RenderApiType foundRenderApiType = RenderApiType::Unknown;
        RenderApiType foundLanguageRenderType = RenderApiType::Unknown;

        for (const auto& arg : args)
        {
            Slang::UnownedStringSlice argSlice = arg.getUnownedSlice();
            if (argSlice.getLength() && argSlice[0] == '-')
            {
                // Look up the rendering API if set
                UnownedStringSlice argName =
                    UnownedStringSlice(argSlice.begin() + 1, argSlice.end());
                RenderApiType renderApiType = RenderApiUtil::findApiTypeByName(argName);

                if (renderApiType != RenderApiType::Unknown)
                {
                    foundRenderApiType = renderApiType;

                    // There should be only one explicit api
                    SLANG_ASSERT(
                        ioRequirements->explicitRenderApi == RenderApiType::Unknown ||
                        ioRequirements->explicitRenderApi == renderApiType);

                    // Set the explicitly set render api
                    ioRequirements->explicitRenderApi = renderApiType;
                    continue;
                }

                // Lookup the target language type
                RenderApiType languageRenderType =
                    RenderApiUtil::findImplicitLanguageRenderApiType(argName);
                if (languageRenderType != RenderApiType::Unknown)
                {
                    foundLanguageRenderType = languageRenderType;

                    // Use the pass thru compiler if these are the sources
                    usePassthru |= (argName == "hlsl" || argName == "glsl");

                    continue;
                }
            }
        }

        // If a render option isn't set use defaultRenderType
        renderApiType = (foundRenderApiType == RenderApiType::Unknown) ? foundLanguageRenderType
                                                                       : foundRenderApiType;
    }

    // The native language for the API
    SlangSourceLanguage nativeLanguage = SLANG_SOURCE_LANGUAGE_UNKNOWN;
    SlangCompileTarget target = SLANG_TARGET_NONE;
    SlangPassThrough passThru = SLANG_PASS_THROUGH_NONE;

    switch (renderApiType)
    {
    case RenderApiType::D3D11:
        target = SLANG_DXBC;
        nativeLanguage = SLANG_SOURCE_LANGUAGE_HLSL;
        passThru = SLANG_PASS_THROUGH_FXC;
        break;
    case RenderApiType::D3D12:
        target = SLANG_DXBC;
        nativeLanguage = SLANG_SOURCE_LANGUAGE_HLSL;
        passThru = SLANG_PASS_THROUGH_FXC;
        if (useDxil)
        {
            target = SLANG_DXIL;
            passThru = SLANG_PASS_THROUGH_DXC;
        }
        break;
    case RenderApiType::Vulkan:
        target = SLANG_SPIRV;
        nativeLanguage = SLANG_SOURCE_LANGUAGE_GLSL;
        passThru = SLANG_PASS_THROUGH_GLSLANG;
        break;
    case RenderApiType::Metal:
        target = SLANG_METAL_LIB;
        nativeLanguage = SLANG_SOURCE_LANGUAGE_METAL;
        passThru = SLANG_PASS_THROUGH_METAL;
        break;
    case RenderApiType::CPU:
        target = SLANG_SHADER_HOST_CALLABLE;
        nativeLanguage = SLANG_SOURCE_LANGUAGE_CPP;
        passThru = SLANG_PASS_THROUGH_GENERIC_C_CPP;
        break;
    case RenderApiType::CUDA:
        target = SLANG_PTX;
        nativeLanguage = SLANG_SOURCE_LANGUAGE_CUDA;
        passThru = SLANG_PASS_THROUGH_NVRTC;
        break;
    case RenderApiType::WebGPU:
        target = SLANG_WGSL;
        nativeLanguage = SLANG_SOURCE_LANGUAGE_WGSL;
        passThru = SLANG_PASS_THROUGH_TINT;
        break;
    }

    SlangSourceLanguage sourceLanguage = nativeLanguage;
    if (!usePassthru)
    {
        sourceLanguage = SLANG_SOURCE_LANGUAGE_SLANG;
        passThru = SLANG_PASS_THROUGH_NONE;
    }

    if (passThru == SLANG_PASS_THROUGH_NONE)
    {
        // Work out backends needed based on the target
        ioRequirements->addUsedBackends(_getPassThroughFlagsForTarget(target));
    }
    else
    {
        ioRequirements->addUsedBackEnd(passThru);
    }

    // Add the render api used
    ioRequirements->addUsedRenderApi(renderApiType);

    return SLANG_OK;
}

static SlangResult _extractSlangCTestRequirements(
    const CommandLine& cmdLine,
    TestRequirements* ioRequirements)
{
    // This determines what the requirements are for a slangc like command line
    // First check pass through
    {
        String passThrough;
        if (SLANG_SUCCEEDED(_extractArg(cmdLine, "-pass-through", passThrough)))
        {
            ioRequirements->addUsedBackEnd(
                TypeTextUtil::findPassThrough(passThrough.getUnownedSlice()));
        }
    }

    // The target if set will also imply a backend
    {
        String targetName;
        if (SLANG_SUCCEEDED(_extractArg(cmdLine, "-target", targetName)))
        {
            const SlangCompileTarget target =
                TypeTextUtil::findCompileTargetFromName(targetName.getUnownedSlice());
            ioRequirements->addUsedBackends(_getPassThroughFlagsForTarget(target));
        }
    }
    return SLANG_OK;
}

static SlangResult _extractReflectionTestRequirements(
    const CommandLine& cmdLine,
    TestRequirements* ioRequirements)
{
    // There are no specialized constraints for a reflection test
    return SLANG_OK;
}

static SlangResult _extractTestRequirements(const CommandLine& cmdLine, TestRequirements* ioInfo)
{
    String exeName = Path::getFileNameWithoutExt(cmdLine.m_executableLocation.m_pathOrName);

    if (exeName == "render-test")
    {
        return _extractRenderTestRequirements(cmdLine, ioInfo);
    }
    else if (exeName == "slangc")
    {
        return _extractSlangCTestRequirements(cmdLine, ioInfo);
    }
    else if (exeName == "slangi")
    {
        return SLANG_OK;
    }
    else if (exeName == "slang-reflection-test")
    {
        return _extractReflectionTestRequirements(cmdLine, ioInfo);
    }

    SLANG_ASSERT(!"Unknown tool type");
    return SLANG_FAIL;
}

static RenderApiFlags _getAvailableRenderApiFlags(TestContext* context)
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    // Only evaluate if it hasn't already been evaluated (the actual evaluation is slow...)
    if (!context->isAvailableRenderApiFlagsValid)
    {
        // Call the render-test tool asking it only to startup a specified render api
        // (taking into account adapter options)

        RenderApiFlags availableRenderApiFlags = 0;
        for (int i = 0; i < int(RenderApiType::CountOf); ++i)
        {
            const RenderApiType apiType = RenderApiType(i);

            if (apiType == RenderApiType::CPU)
            {
                if ((context->availableBackendFlags & PassThroughFlag::Generic_C_CPP) == 0)
                {
                    continue;
                }

                // Check that the session has the generic C/CPP compiler availability - which is all
                // we should need for CPU target
                if (SLANG_SUCCEEDED(context->getSession()->checkPassThroughSupport(
                        SLANG_PASS_THROUGH_GENERIC_C_CPP)))
                {
                    availableRenderApiFlags |= RenderApiFlags(1) << int(apiType);
                }
                continue;
            }

            // See if it's possible the api is available
            if (RenderApiUtil::calcHasApi(apiType))
            {
                if (context->options.skipApiDetection)
                {
                    availableRenderApiFlags |= RenderApiFlags(1) << int(apiType);
                    continue;
                }
                // Try starting up the device
                CommandLine cmdLine;
                cmdLine.setExecutableLocation(
                    ExecutableLocation(context->options.binDir, "render-test"));
                _addRenderTestOptions(context->options, cmdLine);
                // We just want to see if the device can be started up
                cmdLine.addArg("-only-startup");

                // Select what api to use
                StringBuilder builder;
                builder << "-" << RenderApiUtil::getApiName(apiType);
                cmdLine.addArg(builder);
                // Run the render-test tool and see if the device could startup
                ExecuteResult exeRes;
                if (SLANG_SUCCEEDED(
                        spawnAndWaitSharedLibrary(context, "device-startup", cmdLine, exeRes)) &&
                    TestToolUtil::getReturnCodeFromInt(exeRes.resultCode) ==
                        ToolReturnCode::Success)
                {
                    availableRenderApiFlags |= RenderApiFlags(1) << int(apiType);
                    StdWriters::getOut().print(
                        "Check %s: Supported\n",
                        RenderApiUtil::getApiName(apiType).begin());
                }
                else
                {
                    StdWriters::getOut().print(
                        "Check %s: Not Supported\n",
                        RenderApiUtil::getApiName(apiType).begin());
                    const auto out = exeRes.standardOutput;
                    const auto err = exeRes.standardError;
                    if (err.getLength())
                        StdWriters::getOut().print("%s\n", err.getBuffer());
                    if (out.getLength())
                        StdWriters::getOut().print("%s\n", out.getBuffer());
                }
            }
        }

        // After determining available APIs, print adapter info for each one
        if (context->options.showAdapterInfo && availableRenderApiFlags)
        {
            StdWriters::getOut().print("\nAdapter Information for Available APIs:\n");
            for (int i = 0; i < int(RenderApiType::CountOf); ++i)
            {
                const RenderApiType apiType = RenderApiType(i);
                const RenderApiFlags apiFlag = RenderApiFlags(1) << int(apiType);

                if (availableRenderApiFlags & apiFlag)
                {
                    // Create command line to query adapter info
                    CommandLine cmdLine;
                    cmdLine.setExecutableLocation(
                        ExecutableLocation(context->options.binDir, "render-test"));

                    // Add the API type
                    StringBuilder builder;
                    builder << "-" << RenderApiUtil::getApiName(apiType);
                    cmdLine.addArg(builder);

                    // Add flags to show adapter info and only startup
                    cmdLine.addArg("-show-adapter-info");
                    cmdLine.addArg("-only-startup");

                    // Run render-test to get adapter info
                    ExecuteResult exeRes;
                    if (SLANG_SUCCEEDED(
                            spawnAndWaitSharedLibrary(context, "adapter-info", cmdLine, exeRes)))
                    {
                        // Output the adapter info
                        StdWriters::getOut().print(
                            "\n%s:\n%s",
                            RenderApiUtil::getApiName(apiType).begin(),
                            exeRes.standardOutput.getBuffer());
                    }
                }
            }
            StdWriters::getOut().print("\n");
        }

        context->availableRenderApiFlags = availableRenderApiFlags;
        context->isAvailableRenderApiFlagsValid = true;
    }

    return context->availableRenderApiFlags;
}

ToolReturnCode getReturnCode(const ExecuteResult& exeRes)
{
    return TestToolUtil::getReturnCodeFromInt(exeRes.resultCode);
}

ToolReturnCode spawnAndWait(
    TestContext* context,
    const String& testPath,
    SpawnType spawnType,
    const CommandLine& cmdLine,
    ExecuteResult& outExeRes)
{
    if (context->isCollectingRequirements())
    {
        std::lock_guard<std::mutex> lock(context->mutex);
        // If we just want info... don't bother running anything
        const SlangResult res = _extractTestRequirements(cmdLine, context->getTestRequirements());
        // Keep compiler happy on release
        SLANG_UNUSED(res);
        SLANG_ASSERT(SLANG_SUCCEEDED(res));

        return ToolReturnCode::Success;
    }

    const auto& options = context->options;

    const auto finalSpawnType = context->getFinalSpawnType(spawnType);

    SlangResult spawnResult = SLANG_FAIL;
    switch (finalSpawnType)
    {
    case SpawnType::UseExe:
        {
            spawnResult = spawnAndWaitExe(context, testPath, cmdLine, outExeRes);
            break;
        }
    case SpawnType::Default:
    case SpawnType::UseSharedLibrary:
        {
            spawnResult = spawnAndWaitSharedLibrary(context, testPath, cmdLine, outExeRes);
            break;
        }
    case SpawnType::UseFullyIsolatedTestServer:
    case SpawnType::UseTestServer:
        {
            spawnResult =
                spawnAndWaitTestServer(context, finalSpawnType, testPath, cmdLine, outExeRes);
            break;
        }
    default:
        break;
    }

    if (SLANG_FAILED(spawnResult))
    {
        return ToolReturnCode::FailedToRun;
    }

    return getReturnCode(outExeRes);
}

String getOutput(const ExecuteResult& exeRes)
{
    ExecuteResult::ResultCode resultCode = exeRes.resultCode;

    String standardOuptut = exeRes.standardOutput;
    String standardError = exeRes.standardError;

    // We construct a single output string that captures the results
    StringBuilder actualOutputBuilder;
    actualOutputBuilder.append("result code = ");
    actualOutputBuilder.append(resultCode);
    actualOutputBuilder.append("\nstandard error = {\n");
    actualOutputBuilder.append(standardError);
    actualOutputBuilder.append("}\nstandard output = {\n");
    actualOutputBuilder.append(standardOuptut);
    actualOutputBuilder.append("}\n");

    return actualOutputBuilder.produceString();
}

// Finds the specialized or default path for expected data for a test.
// If neither are found, will return an empty string
String findExpectedPath(const TestInput& input, const char* postFix)
{
    StringBuilder specializedBuf;

    // Try the specialized name first
    specializedBuf << input.outputStem;
    if (postFix)
    {
        specializedBuf << postFix;
    }
    if (File::exists(specializedBuf))
    {
        return specializedBuf;
    }


    // Try the default name
    StringBuilder defaultBuf;
    defaultBuf.clear();
    defaultBuf << input.filePath;
    if (postFix)
    {
        defaultBuf << postFix;
    }

    if (File::exists(defaultBuf))
    {
        return defaultBuf;
    }

    // Couldn't find either
    fprintf(
        stderr,
        "referenceOutput '%s' or '%s' not found.\n",
        defaultBuf.getBuffer(),
        specializedBuf.getBuffer());

    return "";
}

static SlangResult _initSlangInterpreter(TestContext* context, CommandLine& ioCmdLine)
{
    ioCmdLine.setExecutableLocation(ExecutableLocation(context->options.binDir, "slangi"));
    return SLANG_OK;
}

static SlangResult _initSlangCompiler(TestContext* context, CommandLine& ioCmdLine)
{
    ioCmdLine.setExecutableLocation(ExecutableLocation(context->options.binDir, "slangc"));

    if (context->options.verbosePaths)
    {
        ioCmdLine.addArgIfNotFound("-verbose-paths");
    }

    // Look for definition of a slot

    {
        const auto prefix = toSlice("-DNV_SHADER_EXTN_SLOT=");

        bool usesNVAPI = false;

        for (auto& arg : ioCmdLine.m_args)
        {
            if (arg.startsWith(prefix))
            {
                // Has NVAPI prefix, meaning
                usesNVAPI = true;
                break;
            }
        }

        // This is necessary because the session can be shared, and the prelude overwritten by the
        // renderer.
        if (usesNVAPI)
        {
            // We want to set the path to NVAPI
            String rootPath;
            SLANG_RETURN_ON_FAIL(TestToolUtil::getRootPath(context->exePath.getBuffer(), rootPath));
            String includePath;
            SLANG_RETURN_ON_FAIL(
                TestToolUtil::getIncludePath(rootPath, "external/nvapi/nvHLSLExtns.h", includePath))

            StringBuilder buf;

            // Include the NVAPI header
            buf << "#include ";

            StringEscapeUtil::appendQuoted(
                StringEscapeUtil::getHandler(StringEscapeUtil::Style::Cpp),
                includePath.getUnownedSlice(),
                buf);
            buf << "\n\n";

            context->getSession()->setLanguagePrelude(SLANG_SOURCE_LANGUAGE_HLSL, buf.getBuffer());
        }
    }

    return SLANG_OK;
}

TestResult asTestResult(ToolReturnCode code)
{
    switch (code)
    {
    case ToolReturnCode::Success:
        return TestResult::Pass;
    case ToolReturnCode::Ignored:
        return TestResult::Ignored;
    default:
        return TestResult::Fail;
    }
}

#define TEST_RETURN_ON_DONE(x)              \
    {                                       \
        const ToolReturnCode toolRet_ = x;  \
        if (TestToolUtil::isDone(toolRet_)) \
        {                                   \
            return asTestResult(toolRet_);  \
        }                                   \
    }

static SlangResult _createArtifactFromHexDump(
    const UnownedStringSlice& hexDump,
    const ArtifactDesc& desc,
    ComPtr<IArtifact>& outArtifact)
{
    // We need to extract the binary
    List<uint8_t> data;
    SLANG_RETURN_ON_FAIL(HexDumpUtil::parseWithMarkers(hexDump, data));

    auto blob = ListBlob::moveCreate(data);
    auto artifact = ArtifactUtil::createArtifact(desc);
    artifact->addRepresentationUnknown(blob);

    outArtifact.swap(artifact);
    return SLANG_OK;
}

static SlangResult _executeBinary(const UnownedStringSlice& hexDump, ExecuteResult& outExeRes)
{
    ComPtr<IArtifact> artifact;
    SLANG_RETURN_ON_FAIL(_createArtifactFromHexDump(
        hexDump,
        ArtifactDesc::make(
            ArtifactKind::Executable,
            ArtifactPayload::HostCPU,
            ArtifactStyle::Unknown),
        artifact));

    ComPtr<IOSFileArtifactRepresentation> fileRep;
    SLANG_RETURN_ON_FAIL(artifact->requireFile(ArtifactKeep::Yes, fileRep.writeRef()));

    const auto fileName = fileRep->getPath();

    // Execute it
    ExecutableLocation exe;
    exe.setPath(fileName);

    CommandLine cmdLine;
    cmdLine.setExecutableLocation(exe);

    return ProcessUtil::execute(cmdLine, outExeRes);
}

static bool _areDiagnosticsEqual(const UnownedStringSlice& a, const UnownedStringSlice& b)
{
    ParseDiagnosticUtil::OutputInfo outA, outB;

    // If we can't parse, we can't match, so fail.
    if (SLANG_FAILED(ParseDiagnosticUtil::parseOutputInfo(a, outA)) ||
        SLANG_FAILED(ParseDiagnosticUtil::parseOutputInfo(b, outB)))
    {
        return false;
    }

    // The result codes must match, and std out
    if (outA.resultCode != outB.resultCode ||
        !StringUtil::areLinesEqual(outA.stdOut.getUnownedSlice(), outB.stdOut.getUnownedSlice()))
    {
        return false;
    }

    // Parse the compiler diagnostics and make sure they are the same.
    // Ignores line number differences
    return ParseDiagnosticUtil::areEqual(
        outA.stdError.getUnownedSlice(),
        outB.stdError.getUnownedSlice(),
        ParseDiagnosticUtil::EqualityFlag::IgnoreLineNos);
}

static bool _areResultsEqual(TestOptions::Type type, const String& a, const String& b)
{
    switch (type)
    {
    case TestOptions::Type::Diagnostic:
        return _areDiagnosticsEqual(a.getUnownedSlice(), b.getUnownedSlice());
    case TestOptions::Type::Normal:
        return a == b;
    default:
        {
            SLANG_ASSERT(!"Unknown test type");
            return false;
        }
    }
}

static String _calcModulePath(const TestInput& input)
{
    // Make the module name the same as the source file
    auto filePath = input.filePath;
    String directory = Path::getParentDirectory(input.outputStem);
    String moduleName = Path::getFileNameWithoutExt(filePath);
    return Path::combine(directory, moduleName);
}

TestResult runDocTest(TestContext* context, TestInput& input)
{
    // need to execute the stand-alone Slang compiler on the file, and compare its output to what we
    // expect
    auto outputStem = input.outputStem;

    CommandLine cmdLine;


    cmdLine.addArg(input.filePath);

    for (auto arg : input.testOptions->args)
    {
        cmdLine.addArg(arg);
    }

    _initSlangCompiler(context, cmdLine);

    ExecuteResult exeRes;
    TEST_RETURN_ON_DONE(spawnAndWait(context, outputStem, input.spawnType, cmdLine, exeRes));

    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }

    String actualOutput = getOutput(exeRes);

    String expectedOutputPath = outputStem + ".expected";
    String expectedOutput;

    // TODO(JS): Might want to check the result code..
    Slang::File::readAllText(expectedOutputPath, expectedOutput);

    // If no expected output file was found, then we
    // expect everything to be empty
    if (expectedOutput.getLength() == 0)
    {
        expectedOutput = "result code = 0\nstandard error = {\n}\nstandard output = {\n}\n";
    }

    TestResult result = TestResult::Pass;

    // Otherwise we compare to the expected output
    if (!_areResultsEqual(input.testOptions->type, expectedOutput, actualOutput))
    {
        context->getTestReporter()->dumpOutputDifference(expectedOutput, actualOutput);
        result = TestResult::Fail;
    }

    // If the test failed, then we write the actual output to a file
    // so that we can easily diff it from the command line and
    // diagnose the problem.
    if (result == TestResult::Fail)
    {
        String actualOutputPath = outputStem + ".actual";
        Slang::File::writeAllText(actualOutputPath, actualOutput);

        context->getTestReporter()->dumpOutputDifference(expectedOutput, actualOutput);
    }

    return result;
}

TestResult runExecutableTest(TestContext* context, TestInput& input)
{
    IDownstreamCompiler* compiler = context->getDefaultCompiler(SLANG_SOURCE_LANGUAGE_CPP);
    if (!compiler)
    {
        return TestResult::Ignored;
    }

    // If we are just collecting requirements, say it passed
    if (context->isCollectingRequirements())
    {
        std::lock_guard<std::mutex> lock(context->mutex);
        context->getTestRequirements()->addUsedBackEnd(SLANG_PASS_THROUGH_GENERIC_C_CPP);
        return TestResult::Pass;
    }

    auto filePath = input.filePath;
    auto outputStem = input.outputStem;

    String actualOutputPath = outputStem + ".actual";
    File::remove(actualOutputPath);

    // Make the module name the same as the current executable path, so it can discover
    // the slang-rt library if needed.
    String modulePath = Path::combine(
        Path::getParentDirectory(Path::getExecutablePath()),
        Path::getFileNameWithoutExt(filePath));

    // String testRoot
    // for(;;)
    // {
    //     String testRoot = Path::getParentDirectory(filePath);
    //     if (testRoot == "")
    //     {
    //         break;
    //     }
    // }
    // printf("test folder = %s\n", testRoot.begin());

    String moduleExePath;
    {
        StringBuilder buf;
        buf << modulePath;
        buf << Process::getExecutableSuffix();
        moduleExePath = buf;
    }

    // Remove the exe if it exists
    File::remove(moduleExePath);

    CommandLine cmdLine;
    _initSlangCompiler(context, cmdLine);

    StringEscapeHandler* escapeHandler =
        StringEscapeUtil::getHandler(StringEscapeUtil::Style::Space);

    List<String> args;
    args.add(filePath);
    args.add("-o");
    args.add(moduleExePath);
    args.add("-target");
    args.add("exe");
    args.add("-Xgenericcpp");
    args.add("-I./include");
    args.add("-Xgenericcpp");
    args.add("-I./external/unordered_dense/include");
    for (auto arg : args)
    {
        // If unescaping is needed, do it
        if (StringEscapeUtil::isUnescapeShellLikeNeeded(escapeHandler, arg.getUnownedSlice()))
        {
            StringBuilder buf;
            StringEscapeUtil::unescapeShellLike(escapeHandler, arg.getUnownedSlice(), buf);
            cmdLine.addArg(buf.produceString());
        }
        else
        {
            cmdLine.addArg(arg);
        }
    }
    ExecuteResult exeRes;

    // TODO(Yong) HACK:
    // Just use shared library now, TestServer spawn mode seems to cause slangc to fail to find its
    // own executable path, and thus failed to find the `gfx.slang` file sitting along side
    // `slangc.exe`. We need to figure out what happened to `Path::getExecutablePath()` inside
    // test-server.
    SpawnType slangcSpawnType = input.spawnType;
    if (slangcSpawnType == SpawnType::UseTestServer)
        slangcSpawnType = SpawnType::UseExe;
    TEST_RETURN_ON_DONE(spawnAndWait(context, outputStem, slangcSpawnType, cmdLine, exeRes));

    String actualOutput;

    // If the actual compilation failed, then the output will be the summary
    if (exeRes.resultCode != 0)
    {
        actualOutput = getOutput(exeRes);
    }
    else
    {
        // Execute the binary and see what we get
        CommandLine cmdLine;

        ExecutableLocation exe;
        exe.setPath(moduleExePath);

        cmdLine.setExecutableLocation(exe);

        ExecuteResult exeRes;
        if (SLANG_FAILED(ProcessUtil::execute(cmdLine, exeRes)))
        {
            return TestResult::Fail;
        }

        // Write the output, and compare to expected
        actualOutput = getOutput(exeRes);
    }

    // Write the output
    Slang::File::writeAllText(actualOutputPath, actualOutput);

    // Check that they are the same
    {
        // Read the expected
        String expectedOutput;

        String expectedOutputPath = outputStem + ".expected";
        Slang::File::readAllText(expectedOutputPath, expectedOutput);

        // Compare if they are the same
        if (!StringUtil::areLinesEqual(
                actualOutput.getUnownedSlice(),
                expectedOutput.getUnownedSlice()))
        {
            context->getTestReporter()->dumpOutputDifference(expectedOutput, actualOutput);
            return TestResult::Fail;
        }
    }

    return TestResult::Pass;
}

TestResult runLanguageServerTest(TestContext* context, TestInput& input)
{
    // We don't support running language server tests in parallel yet.
    std::lock_guard lock(context->mutex);

    if (!context->m_languageServerConnection)
    {
        if (SLANG_FAILED(context->createLanguageServerJSONRPCConnection(
                context->m_languageServerConnection)))
        {
            return TestResult::Fail;
        }
    }
    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }
    auto connection = context->m_languageServerConnection.Ptr();
    LanguageServerProtocol::InitializeParams initParams;
    LanguageServerProtocol::WorkspaceFolder wsFolder;
    wsFolder.name = "test";
    String fullPath;
    Path::getCanonical(input.filePath, fullPath);
    wsFolder.uri = URI::fromLocalFilePath(Path::getParentDirectory(fullPath).getUnownedSlice()).uri;
    initParams.workspaceFolders.add(wsFolder);
    if (SLANG_FAILED(connection->sendCall(
            LanguageServerProtocol::InitializeParams::methodName,
            &initParams,
            JSONValue::makeInt(0))))
    {
        return TestResult::Fail;
    }
    if (SLANG_FAILED(connection->waitForResult(-1)))
    {
        return TestResult::Fail;
    }

    LanguageServerProtocol::InitializeResult initResult;
    if (SLANG_FAILED(connection->getMessage(&initResult)))
    {
        return TestResult::Fail;
    }

    // Send open document call.
    String testFileContent;

    if (SLANG_FAILED(File::readAllText(input.filePath, testFileContent)))
    {
        return TestResult::Fail;
    }

    LanguageServerProtocol::DidOpenTextDocumentParams openDocParams;
    openDocParams.textDocument.version = 0;
    openDocParams.textDocument.uri = URI::fromLocalFilePath(fullPath.getUnownedSlice()).uri;
    openDocParams.textDocument.text = testFileContent;
    connection->sendCall(
        LanguageServerProtocol::DidOpenTextDocumentParams::methodName,
        &openDocParams,
        JSONValue::makeInt(1));
    List<LanguageServerProtocol::PublishDiagnosticsParams> diagnostics;
    bool diagnosticsReceived = false;
    auto waitForNonDiagnosticResponse = [&]() -> SlangResult
    {
        repeat:
            if (SLANG_FAILED(connection->waitForResult(-1)))
                return SLANG_FAIL;
            if (connection->getMessageType() == JSONRPCMessageType::Call)
            {
                JSONRPCCall call;
                connection->getRPC(&call);
                if (call.method == "textDocument/publishDiagnostics")
                {
                    diagnosticsReceived = true;
                    LanguageServerProtocol::PublishDiagnosticsParams arg;
                    if (SLANG_FAILED(connection->getMessage(&arg)))
                        return SLANG_FAIL;
                    diagnostics.add(arg);
                    goto repeat;
                }
            }
            return SLANG_OK;
    };

    List<UnownedStringSlice> lines;
    StringUtil::calcLines(testFileContent.getUnownedSlice(), lines);

    StringBuilder actualOutputSB;
    auto parseLocation = [&](UnownedStringSlice text, Index startPos, Int& linePos, Int& colPos)
    {
        linePos = StringUtil::parseIntAndAdvancePos(text.trimStart(), startPos);
        startPos++;
        colPos = StringUtil::parseIntAndAdvancePos(text.trimStart(), startPos);
        return startPos;
    };
    int callId = 2;
    for (auto line : lines)
    {
        line = line.trimStart();
        if (!line.startsWith("//"))
            continue;
        line = line.tail(2).trimStart();
        if (line.startsWith("COMPLETE:"))
        {
            auto arg = line.tail(UnownedStringSlice("COMPLETE:").getLength());
            Int linePos, colPos;
            parseLocation(arg, 0, linePos, colPos);

            LanguageServerProtocol::CompletionParams params;
            params.position.line = int(linePos - 1);
            params.position.character = int(colPos - 1);
            params.textDocument.uri = openDocParams.textDocument.uri;
            if (SLANG_FAILED(connection->sendCall(
                    LanguageServerProtocol::CompletionParams::methodName,
                    &params,
                    JSONValue::makeInt(callId++))))
            {
                return TestResult::Fail;
            }
            if (SLANG_FAILED(waitForNonDiagnosticResponse()))
                return TestResult::Fail;
            actualOutputSB << "--------\n";
            LanguageServerProtocol::NullResponse nullResponse;
            List<LanguageServerProtocol::CompletionItem> completionItems;
            if (SLANG_SUCCEEDED(connection->getMessage(&nullResponse)))
            {
                actualOutputSB << "null\n";
            }
            else if (SLANG_SUCCEEDED(connection->getMessage(&completionItems)))
            {
                for (auto item : completionItems)
                {
                    actualOutputSB << item.label << ": " << item.kind << " " << item.detail << " ";
                    for (auto ch : item.commitCharacters)
                        actualOutputSB << ch;
                    actualOutputSB << "\n";
                }
            }
        }
        else if (line.startsWith("SIGNATURE:"))
        {
            auto arg = line.tail(UnownedStringSlice("SIGNATURE:").getLength());
            Int linePos, colPos;
            parseLocation(arg, 0, linePos, colPos);

            LanguageServerProtocol::SignatureHelpParams params;
            params.position.line = int(linePos - 1);
            params.position.character = int(colPos - 1);
            params.textDocument.uri = openDocParams.textDocument.uri;
            if (SLANG_FAILED(connection->sendCall(
                    LanguageServerProtocol::SignatureHelpParams::methodName,
                    &params,
                    JSONValue::makeInt(callId++))))
            {
                return TestResult::Fail;
            }
            if (SLANG_FAILED(waitForNonDiagnosticResponse()))
                return TestResult::Fail;
            actualOutputSB << "--------\n";
            LanguageServerProtocol::NullResponse nullResponse;
            LanguageServerProtocol::SignatureHelp sigInfo;
            if (SLANG_SUCCEEDED(connection->getMessage(&nullResponse)))
            {
                actualOutputSB << "null\n";
            }
            else if (SLANG_SUCCEEDED(connection->getMessage(&sigInfo)))
            {
                actualOutputSB << "activeParameter: " << sigInfo.activeParameter << "\n";
                actualOutputSB << "activeSignature: " << sigInfo.activeSignature << "\n";
                for (auto item : sigInfo.signatures)
                {
                    actualOutputSB << item.label << ":";
                    for (auto param : item.parameters)
                    {
                        actualOutputSB << " (" << param.label[0] << "," << param.label[1] << ")";
                    }
                    actualOutputSB << "\n";
                    actualOutputSB << item.documentation.value << "\n";
                }
            }
        }
        else if (line.startsWith("HOVER:"))
        {
            auto arg = line.tail(UnownedStringSlice("HOVER:").getLength());
            Int linePos, colPos;
            parseLocation(arg, 0, linePos, colPos);

            LanguageServerProtocol::HoverParams params;
            params.position.line = int(linePos - 1);
            params.position.character = int(colPos - 1);
            params.textDocument.uri = openDocParams.textDocument.uri;
            if (SLANG_FAILED(connection->sendCall(
                    LanguageServerProtocol::HoverParams::methodName,
                    &params,
                    JSONValue::makeInt(callId++))))
            {
                return TestResult::Fail;
            }
            if (SLANG_FAILED(waitForNonDiagnosticResponse()))
                return TestResult::Fail;
            actualOutputSB << "--------\n";
            LanguageServerProtocol::NullResponse nullResponse;
            LanguageServerProtocol::Hover hover;
            if (SLANG_SUCCEEDED(connection->getMessage(&nullResponse)))
            {
                actualOutputSB << "null\n";
            }
            else if (SLANG_SUCCEEDED(connection->getMessage(&hover)))
            {
                actualOutputSB << "range: " << hover.range.start.line << ","
                               << hover.range.start.character << " - " << hover.range.end.line
                               << "," << hover.range.end.character;
                actualOutputSB << "\ncontent:\n" << hover.contents.value << "\n";
            }
        }
        else if (line.startsWith("DIAGNOSTICS"))
        {
            if (!diagnosticsReceived)
            {
                waitForNonDiagnosticResponse();
            }
            actualOutputSB << "--------\n";
            for (auto item : diagnostics)
            {
                actualOutputSB << item.uri << "\n";
                for (auto msg : item.diagnostics)
                {
                    actualOutputSB << msg.range.start.line << "," << msg.range.start.character
                                   << "-" << msg.range.end.line << "," << msg.range.end.character
                                   << " " << msg.message;
                }
            }
        }
    }
    LanguageServerProtocol::DidCloseTextDocumentParams closeDocParams;
    closeDocParams.textDocument.uri = URI::fromLocalFilePath(fullPath.getUnownedSlice()).uri;
    connection->sendCall(
        LanguageServerProtocol::DidCloseTextDocumentParams::methodName,
        &closeDocParams,
        JSONValue::makeInt(1));

    auto outputStem = input.outputStem;
    String expectedOutputPath = outputStem + ".expected.txt";
    String expectedOutput;

    Slang::File::readAllText(expectedOutputPath, expectedOutput);
    expectedOutput = expectedOutput.trim();

    TestResult result = TestResult::Pass;

    auto actualOutput = actualOutputSB.produceString();

    // Redact absolute file names from actualOutput
    List<UnownedStringSlice> outputLines;
    StringUtil::calcLines(actualOutput.getUnownedSlice(), outputLines);
    StringBuilder redactedSB;
    for (auto line : outputLines)
    {
        Index extIdx = line.indexOf(UnownedStringSlice(".slang"));
        if (extIdx == -1)
        {
            redactedSB << line << "\n";
            continue;
        }
        redactedSB << "{REDACTED}" << line.tail(extIdx) << "\n";
    }

    actualOutput = redactedSB.produceString().trim();

    String fileCheckPrefix;
    const bool isFileCheckTest = input.testOptions->getFileCheckPrefix(fileCheckPrefix);
    if (isFileCheckTest)
    {
        result = _fileCheckTest(*context, input.filePath, fileCheckPrefix, actualOutput);
    }
    else
    {
        if (!_areResultsEqual(input.testOptions->type, expectedOutput, actualOutput))
        {
            if (expectedOutput.startsWith("CONTAINS"))
            {
                List<UnownedStringSlice> words;
                List<UnownedStringSlice> expectedLines;
                StringUtil::calcLines(expectedOutput.getUnownedSlice(), expectedLines);
                if (expectedLines.getCount() >= 1)
                {
                    StringUtil::split(expectedLines[0], ' ', words);
                    if (words.getCount() >= 2)
                    {
                        if (actualOutput.contains(words[1].trim()))
                        {
                            return result;
                        }
                    }
                }
            }
            context->getTestReporter()->dumpOutputDifference(expectedOutput, actualOutput);
            result = TestResult::Fail;
        }
    }

    // If the test failed, then we write the actual output to a file
    // so that we can easily diff it from the command line and
    // diagnose the problem.
    if (result == TestResult::Fail)
    {
        String actualOutputPath = outputStem + ".actual";
        Slang::File::writeAllText(actualOutputPath, actualOutput);

        context->getTestReporter()->dumpOutputDifference(expectedOutput, actualOutput);
    }
    return result;
}

TestResult runSimpleTest(TestContext* context, TestInput& input)
{
    // need to execute the stand-alone Slang compiler on the file, and compare its output to what we
    // expect
    auto outputStem = input.outputStem;

    CommandLine cmdLine;

    if (input.testOptions->command != "SIMPLE_EX")
    {
        cmdLine.addArg(input.filePath);
    }

    for (auto arg : input.testOptions->args)
    {
        cmdLine.addArg(arg);
    }

    // If we can't set up for simple compilation, it's because some external resource isn't
    // available such as NVAPI headers. In that case we just ignore the test.
    if (SLANG_FAILED(_initSlangCompiler(context, cmdLine)))
    {
        return TestResult::Ignored;
    }

    ExecuteResult exeRes;
    TEST_RETURN_ON_DONE(spawnAndWait(context, outputStem, input.spawnType, cmdLine, exeRes));

    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }

    // See what kind of target it is
    SlangCompileTarget target = SLANG_TARGET_UNKNOWN;
    {
        const auto& args = input.testOptions->args;
        const Index targetIndex = args.indexOf("-target");
        if (targetIndex != Index(-1) && targetIndex + 1 < args.getCount())
        {
            target =
                TypeTextUtil::findCompileTargetFromName(args[targetIndex + 1].getUnownedSlice());
        }
    }

    // If it's executable we run it and use it's output
    if (target == SLANG_HOST_EXECUTABLE)
    {
        ExecuteResult runExeRes;
        if (SLANG_FAILED(_executeBinary(exeRes.standardOutput.getUnownedSlice(), runExeRes)))
        {
            return TestResult::Fail;
        }
        exeRes = runExeRes;
    }

    String actualOutput = getOutput(exeRes);

    return _validateOutput(
        context,
        input,
        actualOutput,
        false,
        "result code = 0\nstandard error = {\n}\nstandard output = {\n}\n",
        [&input](auto e, auto a) { return _areResultsEqual(input.testOptions->type, e, a); });
}

TestResult runSimpleLineTest(TestContext* context, TestInput& input)
{
    // need to execute the stand-alone Slang compiler on the file, and compare its output to what we
    // expect
    auto outputStem = input.outputStem;

    CommandLine cmdLine;
    _initSlangCompiler(context, cmdLine);

    cmdLine.addArg(input.filePath);

    for (auto arg : input.testOptions->args)
    {
        cmdLine.addArg(arg);
    }

    ExecuteResult exeRes;
    TEST_RETURN_ON_DONE(spawnAndWait(context, outputStem, input.spawnType, cmdLine, exeRes));

    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }

    // Parse all the diagnostics so we can extract line numbers
    auto diagnostics = ArtifactDiagnostics::create();
    if (SLANG_FAILED(ParseDiagnosticUtil::parseDiagnostics(
            exeRes.standardError.getUnownedSlice(),
            diagnostics)) ||
        diagnostics->getCount() <= 0)
    {
        // Write out the diagnostics which couldn't be parsed.

        String actualOutputPath = outputStem + ".actual";
        Slang::File::writeAllText(actualOutputPath, exeRes.standardError);

        return TestResult::Fail;
    }

    StringBuilder actualOutput;

    if (diagnostics->getCount() > 0)
    {
        actualOutput << diagnostics->getAt(0)->location.line << "\n";
    }
    else
    {
        actualOutput << "No output diagnostics\n";
    }

    return _validateOutput(context, input, actualOutput, false);
}

TestResult runInterpreterTest(TestContext* context, TestInput& input)
{
    // need to execute the stand-alone Slang compiler on the file, and compare its output to what we
    // expect
    auto outputStem = input.outputStem;

    CommandLine cmdLine;

    List<String> args;

    for (Index i = 0; i < input.testOptions->args.getCount(); i++)
    {
        auto& arg = input.testOptions->args[i];
        if (arg == "-disasm")
            cmdLine.addArg(arg);
        else if (arg == "-entry")
        {
            cmdLine.addArg(arg);
            i++;
            if (i < input.testOptions->args.getCount())
            {
                cmdLine.addArg(input.testOptions->args[i]);
            }
        }
        else
        {
            args.add(arg);
        }
    }

    cmdLine.addArg(input.filePath);

    for (auto arg : args)
    {
        cmdLine.addArg(arg);
    }

    if (SLANG_FAILED(_initSlangInterpreter(context, cmdLine)))
    {
        return TestResult::Ignored;
    }

    ExecuteResult exeRes;
    TEST_RETURN_ON_DONE(spawnAndWait(context, outputStem, input.spawnType, cmdLine, exeRes));

    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }

    String actualOutput = getOutput(exeRes);

    return _validateOutput(
        context,
        input,
        actualOutput,
        false,
        "result code = 0\nstandard error = {\n}\nstandard output = {\n}\n",
        [&input](auto e, auto a) { return _areResultsEqual(input.testOptions->type, e, a); });
}

TestResult runCompile(TestContext* context, TestInput& input)
{
    auto outputStem = input.outputStem;

    CommandLine cmdLine;
    _initSlangCompiler(context, cmdLine);

    StringEscapeHandler* escapeHandler =
        StringEscapeUtil::getHandler(StringEscapeUtil::Style::Space);

    for (auto arg : input.testOptions->args)
    {
        // If unescaping is needed, do it
        if (StringEscapeUtil::isUnescapeShellLikeNeeded(escapeHandler, arg.getUnownedSlice()))
        {
            StringBuilder buf;
            StringEscapeUtil::unescapeShellLike(escapeHandler, arg.getUnownedSlice(), buf);
            cmdLine.addArg(buf.produceString());
        }
        else
        {
            cmdLine.addArg(arg);
        }
    }

    ExecuteResult exeRes;
    TEST_RETURN_ON_DONE(spawnAndWait(context, outputStem, input.spawnType, cmdLine, exeRes));
    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }

    if (exeRes.resultCode != 0)
    {
        auto reporter = context->getTestReporter();
        if (reporter)
        {
            auto output = getOutput(exeRes);
            reporter->message(TestMessageType::TestFailure, output);
        }

        return TestResult::Fail;
    }

    return TestResult::Pass;
}

TestResult runSimpleCompareCommandLineTest(TestContext* context, TestInput& input)
{
    TestInput workInput(input);
    // Use the original files input to compare with
    workInput.outputStem = input.filePath;
    // Force to using exes
    workInput.spawnType = SpawnType::UseExe;

    return runSimpleTest(context, workInput);
}

static SlangResult _parseJSON(
    const UnownedStringSlice& slice,
    DiagnosticSink* sink,
    JSONContainer* container,
    JSONValue& outValue)
{
    SourceManager* sourceManager = sink->getSourceManager();

    SourceFile* sourceFile =
        sourceManager->createSourceFileWithString(PathInfo::makeUnknown(), slice);
    SourceView* sourceView = sourceManager->createSourceView(sourceFile, nullptr, SourceLoc());

    JSONLexer lexer;
    lexer.init(sourceView, sink);

    JSONBuilder builder(container);

    JSONParser parser;
    SLANG_RETURN_ON_FAIL(parser.parse(&lexer, sourceView, &builder, sink));

    outValue = builder.getRootValue();
    return SLANG_OK;
}

TestResult runReflectionTest(TestContext* context, TestInput& input)
{
    const auto& options = context->options;
    const auto& filePath = input.filePath;
    auto& outputStem = input.outputStem;

    bool isCPUTest = input.testOptions->command.startsWith("CPU_");

    CommandLine cmdLine;

    cmdLine.setExecutableLocation(ExecutableLocation(options.binDir, "slang-reflection-test"));
    cmdLine.addArg(filePath);

    for (auto arg : input.testOptions->args)
    {
        cmdLine.addArg(arg);
    }

    ExecuteResult exeRes;
    TEST_RETURN_ON_DONE(spawnAndWait(context, outputStem, input.spawnType, cmdLine, exeRes));

    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }

    String actualOutput = getOutput(exeRes);

    if (isCPUTest)
    {
#if SLANG_PTR_IS_32
        outputStem.append(".32");
#else
        outputStem.append(".64");
#endif
    }

    // Extrac the stand
    ParseDiagnosticUtil::OutputInfo outputInfo;
    if (SLANG_SUCCEEDED(
            ParseDiagnosticUtil::parseOutputInfo(actualOutput.getUnownedSlice(), outputInfo)))
    {
        const auto toolReturnCode = ToolReturnCode(outputInfo.resultCode);

        // The output should be JSON.
        // Parse it to check that it is valid json
        if (toolReturnCode == ToolReturnCode::Success)
        {
            SourceManager sourceManager;
            sourceManager.initialize(nullptr, nullptr);

            JSONContainer container(&sourceManager);

            DiagnosticSink sink;
            sink.init(&sourceManager, nullptr);

            JSONValue value;
            if (SLANG_FAILED(
                    _parseJSON(outputInfo.stdOut.getUnownedSlice(), &sink, &container, value)))
            {
                // Unable to parse as JSON

                context->getTestReporter()->messageFormat(
                    TestMessageType::RunError,
                    "Unable to parse reflection JSON '%s'\n",
                    input.outputStem.getBuffer());

                String actualOutputPath = input.outputStem + ".actual";
                Slang::File::writeAllText(actualOutputPath, actualOutput);
                return TestResult::Fail;
            }
        }
    }

    return _validateOutput(context, input, actualOutput);
}

static String _calcSummary(IArtifactDiagnostics* inDiagnostics)
{
    auto diagnostics = cloneInterface(inDiagnostics);

    // We only want to analyze errors for now
    diagnostics->removeBySeverity(ArtifactDiagnostic::Severity::Info);
    diagnostics->removeBySeverity(ArtifactDiagnostic::Severity::Warning);

    ComPtr<ISlangBlob> summary;
    diagnostics->calcSimplifiedSummary(summary.writeRef());

    return StringUtil::getString(summary);
}

static TestResult runCPPCompilerCompile(TestContext* context, TestInput& input)
{
    IDownstreamCompiler* compiler = context->getDefaultCompiler(SLANG_SOURCE_LANGUAGE_CPP);
    if (!compiler)
    {
        return TestResult::Ignored;
    }

    // need to execute the stand-alone Slang compiler on the file, and compare its output to what we
    // expect

    auto outputStem = input.outputStem;

    CommandLine cmdLine;
    _initSlangCompiler(context, cmdLine);

    cmdLine.addArg(input.filePath);
    for (auto arg : input.testOptions->args)
    {
        cmdLine.addArg(arg);
    }

    ExecuteResult exeRes;
    TEST_RETURN_ON_DONE(spawnAndWait(context, outputStem, input.spawnType, cmdLine, exeRes));
    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }

    // Dump out what happened
    {
        String actualOutputPath = outputStem + ".actual";
        Slang::File::writeAllText(actualOutputPath, getOutput(exeRes));
    }

    if (exeRes.resultCode != 0)
    {
        return TestResult::Fail;
    }

    return TestResult::Pass;
}

static TestResult runCPPCompilerSharedLibrary(TestContext* context, TestInput& input)
{
    IDownstreamCompiler* compiler = context->getDefaultCompiler(SLANG_SOURCE_LANGUAGE_CPP);
    if (!compiler)
    {
        std::lock_guard<std::mutex> lock(context->mutex);
        return TestResult::Ignored;
    }

    // If we are just collecting requirements, say it passed
    if (context->isCollectingRequirements())
    {
        context->getTestRequirements()->addUsedBackEnd(SLANG_PASS_THROUGH_GENERIC_C_CPP);
        return TestResult::Pass;
    }

    auto outputStem = input.outputStem;
    auto filePath = input.filePath;

    String actualOutputPath = outputStem + ".actual";
    File::remove(actualOutputPath);

    // Make the module name the same as the source file
    String modulePath = _calcModulePath(input);
    String ext = Path::getPathExt(filePath);

    // Remove the binary..
    String sharedLibraryPath = SharedLibrary::calcPlatformPath(modulePath.getUnownedSlice());
    File::remove(sharedLibraryPath);

    // Set up the compilation options
    DownstreamCompileOptions options;

    options.sourceLanguage = (ext == "c") ? SLANG_SOURCE_LANGUAGE_C : SLANG_SOURCE_LANGUAGE_CPP;

    // Build a shared library
    options.targetType = SLANG_SHADER_SHARED_LIBRARY;

    auto helper = DefaultArtifactHelper::getSingleton();

    // Compile this source
    ComPtr<IArtifact> sourceArtifact;

    // If set, we store the artifact in memory without a name.
    bool checkMemory = false;
    if (checkMemory)
    {
        helper->createArtifact(
            ArtifactDescUtil::makeDescForSourceLanguage(options.sourceLanguage),
            "",
            sourceArtifact.writeRef());

        ComPtr<IOSFileArtifactRepresentation> fileRep;
        // Let's just add a blob with the contents
        helper->createOSFileArtifactRepresentation(
            IOSFileArtifactRepresentation::Kind::Reference,
            asCharSlice(filePath.getUnownedSlice()),
            nullptr,
            fileRep.writeRef());

        ComPtr<ICastable> castable;
        fileRep->createRepresentation(ISlangBlob::getTypeGuid(), castable.writeRef());

        sourceArtifact->addRepresentation(castable);
    }
    else
    {
        helper->createOSFileArtifact(
            ArtifactDescUtil::makeDescForSourceLanguage(options.sourceLanguage),
            asCharSlice(filePath.getUnownedSlice()),
            sourceArtifact.writeRef());
    }

    TerminatedCharSlice includePaths[] = {TerminatedCharSlice(".")};

    options.sourceArtifacts = makeSlice(sourceArtifact.readRef(), 1);
    options.includePaths = makeSlice(includePaths, SLANG_COUNT_OF(includePaths));
    options.modulePath = SliceUtil::asTerminatedCharSlice(modulePath);

    ComPtr<IArtifact> artifact;
    if (SLANG_FAILED(compiler->compile(options, artifact.writeRef())))
    {
        return TestResult::Fail;
    }

    auto diagnostics = findAssociatedRepresentation<IArtifactDiagnostics>(artifact);

    if (diagnostics && SLANG_FAILED(diagnostics->getResult()))
    {
        // Compilation failed
        String actualOutput = _calcSummary(diagnostics);

        // Write the output
        Slang::File::writeAllText(actualOutputPath, actualOutput);

        // Check that they are the same
        {
            // Read the expected
            String expectedOutput;

            String expectedOutputPath = outputStem + ".expected";
            Slang::File::readAllText(expectedOutputPath, expectedOutput);

            // Compare if they are the same
            if (!StringUtil::areLinesEqual(
                    actualOutput.getUnownedSlice(),
                    expectedOutput.getUnownedSlice()))
            {
                context->getTestReporter()->dumpOutputDifference(expectedOutput, actualOutput);
                return TestResult::Fail;
            }
        }
    }
    else
    {
        SharedLibrary::Handle handle;
        if (SLANG_FAILED(
                SharedLibrary::loadWithPlatformPath(sharedLibraryPath.getBuffer(), handle)))
        {
            return TestResult::Fail;
        }

        const int inValue = 10;
        const char inBuffer[] = "Hello World!";

        char buffer[128] = "";
        int value = 0;

        typedef int (*TestFunc)(int intValue, const char* textValue, char* outTextValue);

        // We could capture output if we passed in a ISlangWriter - but for that to work we'd need a
        TestFunc testFunc = (TestFunc)SharedLibrary::findSymbolAddressByName(handle, "test");
        if (testFunc)
        {
            value = testFunc(inValue, inBuffer, buffer);
        }
        else
        {
            printf("Unable to access 'test' function\n");
        }

        SharedLibrary::unload(handle);

        if (!(inValue == value && strcmp(inBuffer, buffer) == 0))
        {
            return TestResult::Fail;
        }
    }

    return TestResult::Pass;
}

static TestResult runCPPCompilerExecute(TestContext* context, TestInput& input)
{
    IDownstreamCompiler* compiler = context->getDefaultCompiler(SLANG_SOURCE_LANGUAGE_CPP);
    if (!compiler)
    {
        return TestResult::Ignored;
    }

    // If we are just collecting requirements, say it passed
    if (context->isCollectingRequirements())
    {
        std::lock_guard<std::mutex> lock(context->mutex);
        context->getTestRequirements()->addUsedBackEnd(SLANG_PASS_THROUGH_GENERIC_C_CPP);
        return TestResult::Pass;
    }

    auto filePath = input.filePath;
    auto outputStem = input.outputStem;

    String actualOutputPath = outputStem + ".actual";
    File::remove(actualOutputPath);

    // Make the module name the same as the source file
    String ext = Path::getPathExt(filePath);
    String modulePath = _calcModulePath(input);

    // Remove the binary..
    String moduleExePath;
    {
        StringBuilder buf;
        buf << modulePath;
        buf << Process::getExecutableSuffix();
        moduleExePath = buf;
    }

    // Remove the exe if it exists
    File::remove(moduleExePath);

    // Set up the compilation options
    DownstreamCompileOptions options;

    options.sourceLanguage = (ext == "c") ? SLANG_SOURCE_LANGUAGE_C : SLANG_SOURCE_LANGUAGE_CPP;

    TerminatedCharSlice filePaths[] = {SliceUtil::asTerminatedCharSlice(filePath)};

    auto helper = DefaultArtifactHelper::getSingleton();

    ComPtr<IArtifact> sourceArtifact;
    helper->createOSFileArtifact(
        ArtifactDescUtil::makeDescForSourceLanguage(options.sourceLanguage),
        asCharSlice(filePath.getUnownedSlice()),
        sourceArtifact.writeRef());

    // Compile this source
    options.sourceArtifacts = makeSlice(sourceArtifact.readRef(), 1);
    options.modulePath = SliceUtil::asTerminatedCharSlice(modulePath);

    ComPtr<IArtifact> artifact;
    if (SLANG_FAILED(compiler->compile(options, artifact.writeRef())))
    {
        return TestResult::Fail;
    }

    String actualOutput;

    auto diagnostics = findAssociatedRepresentation<IArtifactDiagnostics>(artifact);

    // If the actual compilation failed, then the output will be the summary
    if (diagnostics && SLANG_FAILED(diagnostics->getResult()))
    {
        actualOutput = _calcSummary(diagnostics);
    }
    else
    {
        // Execute the binary and see what we get
        CommandLine cmdLine;

        ExecutableLocation exe;
        exe.setPath(moduleExePath);

        cmdLine.setExecutableLocation(exe);

        ExecuteResult exeRes;
        if (SLANG_FAILED(ProcessUtil::execute(cmdLine, exeRes)))
        {
            return TestResult::Fail;
        }

        // Write the output, and compare to expected
        actualOutput = getOutput(exeRes);
    }

    // Write the output
    Slang::File::writeAllText(actualOutputPath, actualOutput);

    // Check that they are the same
    {
        // Read the expected
        String expectedOutput;

        String expectedOutputPath = outputStem + ".expected";
        Slang::File::readAllText(expectedOutputPath, expectedOutput);

        // Compare if they are the same
        if (!StringUtil::areLinesEqual(
                actualOutput.getUnownedSlice(),
                expectedOutput.getUnownedSlice()))
        {
            context->getTestReporter()->dumpOutputDifference(expectedOutput, actualOutput);
            return TestResult::Fail;
        }
    }

    return TestResult::Pass;
}

// Returns TestResult::Ignored if we don't have the capability to run the passthrough compiler
// Returns TestResult::Fail if we can't write the expected output debug file
// Otherwise return TestResult::Pass and if we are not just collecting
// requirements, writes the output into the `expectedOutput` parameter
static TestResult generateExpectedOutput(
    TestContext* const context,
    const TestInput& input,
    String& expectedOutput)
{
    auto filePath = input.filePath;
    auto outputStem = input.outputStem;

    CommandLine expectedCmdLine;

    _initSlangCompiler(context, expectedCmdLine);

    const auto& args = input.testOptions->args;

    const Index targetIndex = args.indexOf("-target");
    if (targetIndex != Index(-1) && targetIndex + 1 < args.getCount())
    {
        const SlangCompileTarget target =
            TypeTextUtil::findCompileTargetFromName(args[targetIndex + 1].getUnownedSlice());

        // Check the session supports it. If not we ignore it
        if (SLANG_FAILED(context->getSession()->checkCompileTargetSupport(target)))
        {
            return TestResult::Ignored;
        }

        switch (target)
        {
        case SLANG_DXIL:
        case SLANG_DXIL_ASM:
            {
                expectedCmdLine.addArg(filePath + ".hlsl");
                expectedCmdLine.addArg("-pass-through");
                expectedCmdLine.addArg("dxc");
                break;
            }
        case SLANG_DXBC:
        case SLANG_DXBC_ASM:
            {
                expectedCmdLine.addArg(filePath + ".hlsl");
                expectedCmdLine.addArg("-pass-through");
                expectedCmdLine.addArg("fxc");
                break;
            }
        default:
            {
                expectedCmdLine.addArg(filePath + ".glsl");
                expectedCmdLine.addArg("-emit-spirv-via-glsl");
                expectedCmdLine.addArg("-pass-through");
                expectedCmdLine.addArg("glslang");
                break;
            }
        }
    }

    for (auto arg : args)
    {
        expectedCmdLine.addArg(arg);
    }

    ExecuteResult expectedExeRes;
    TEST_RETURN_ON_DONE(
        spawnAndWait(context, outputStem, input.spawnType, expectedCmdLine, expectedExeRes));

    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }

    expectedOutput = getOutput(expectedExeRes);
    String expectedOutputPath = outputStem + ".expected";

    if (SLANG_FAILED(Slang::File::writeAllText(expectedOutputPath, expectedOutput)))
    {
        context->getTestReporter()->messageFormat(
            TestMessageType::TestFailure,
            "Failed to write test expected output to %s",
            expectedOutputPath.getBuffer());
        return TestResult::Fail;
    }

    return TestResult::Pass;
}

// Returns TestResult::Fail if compilation fails
// Otherwise return TestResult::Pass and if we are not just collecting
// requirements, writes the output into the `expectedOutput` parameter
TestResult generateActualOutput(
    TestContext* const context,
    const TestInput& input,
    String& actualOutput)
{
    auto filePath = input.filePath;

    CommandLine actualCmdLine;
    _initSlangCompiler(context, actualCmdLine);
    actualCmdLine.addArg(filePath);
    actualCmdLine.addArg("-emit-spirv-via-glsl");

    const auto& args = input.testOptions->args;

    for (auto arg : input.testOptions->args)
    {
        actualCmdLine.addArg(arg);
    }

    ExecuteResult actualExeRes;
    TEST_RETURN_ON_DONE(
        spawnAndWait(context, input.outputStem, input.spawnType, actualCmdLine, actualExeRes));

    // Early out if we're just collecting requirements
    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }

    actualOutput = getOutput(actualExeRes);

    // Always fail if the compilation produced a failure, just
    // to catch situations where, e.g., command-line options parsing
    // caused the same error in both the Slang and glslang cases.
    //
    if (actualExeRes.resultCode != 0)
    {
        return TestResult::Fail;
    }

    return TestResult::Pass;
}

TestResult runCrossCompilerTest(TestContext* context, TestInput& input)
{
    // Need to execute the stand-alone Slang compiler on the file
    // then on the same file + `.glsl` and compare output
    //
    // Or, in the case of a filecheck test, instead of comparing against the
    // +".glsl" version, we run some filecheck rules on it

    String fileCheckPrefix;
    const bool isFileCheckTest = input.testOptions->getFileCheckPrefix(fileCheckPrefix);

    String actualOutput;
    if (TestResult r = generateActualOutput(context, input, actualOutput); r != TestResult::Pass)
    {
        return r;
    }

    // Only generate the expected output if this is a comparison against some
    // known-good glsl/hlsl input
    String expectedOutput;
    if (!isFileCheckTest)
    {
        if (TestResult r = generateExpectedOutput(context, input, expectedOutput);
            r != TestResult::Pass)
        {
            return r;
        }
    }

    // Early out if we're just collecting requirements
    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }

    TestResult result = TestResult::Pass;

    if (isFileCheckTest)
    {
        result = _fileCheckTest(*context, input.filePath, fileCheckPrefix, actualOutput);
        // TODO: It might be a good idea to sanity check any expected output
        // source files against the filecheck rules if they're applicable.
        //
        // Something like:
        // fileCheckTest(context, prefix="HLSL", input, filePath + ".hlsl");
    }
    else
    {
        if (!StringUtil::areLinesEqual(
                actualOutput.getUnownedSlice(),
                expectedOutput.getUnownedSlice()))
        {
            result = TestResult::Fail;
            context->getTestReporter()->dumpOutputDifference(expectedOutput, actualOutput);
        }
    }

    // If the test failed, then we write the actual output to a file
    // so that we can easily inspect it from the command line and
    // diagnose the problem.
    if (result == TestResult::Fail)
    {
        String actualOutputPath = input.outputStem + ".actual";
        Slang::File::writeAllText(actualOutputPath, actualOutput);
    }

    return result;
}

TestResult generateHLSLBaseline(
    TestContext* context,
    TestInput& input,
    char const* targetFormat,
    char const* passThroughName)
{
    auto filePath999 = input.filePath;
    auto outputStem = input.outputStem;

    CommandLine cmdLine;
    _initSlangCompiler(context, cmdLine);

    cmdLine.addArg(filePath999);

    for (auto arg : input.testOptions->args)
    {
        cmdLine.addArg(arg);
    }

    cmdLine.addArg("-target");
    cmdLine.addArg(targetFormat);
    cmdLine.addArg("-pass-through");
    cmdLine.addArg(passThroughName);

    ExecuteResult exeRes;
    TEST_RETURN_ON_DONE(spawnAndWait(context, outputStem, input.spawnType, cmdLine, exeRes));

    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }

    String expectedOutput = getOutput(exeRes);
    String expectedOutputPath = outputStem + ".expected";

    if (SLANG_FAILED(Slang::File::writeAllText(expectedOutputPath, expectedOutput)))
    {
        return TestResult::Fail;
    }

    return TestResult::Pass;
}

TestResult generateHLSLBaseline(TestContext* context, TestInput& input)
{
    return generateHLSLBaseline(context, input, "dxbc-assembly", "fxc");
}

static TestResult _runHLSLComparisonTest(
    TestContext* context,
    TestInput& input,
    char const* targetFormat,
    char const* passThroughName)
{
    auto filePath999 = input.filePath;
    auto outputStem = input.outputStem;

    // We will use the Microsoft compiler to generate out expected output here
    String expectedOutputPath = outputStem + ".expected";

    // Generate the expected output using standard HLSL compiler
    generateHLSLBaseline(context, input, targetFormat, passThroughName);

    // need to execute the stand-alone Slang compiler on the file, and compare its output to what we
    // expect

    CommandLine cmdLine;
    _initSlangCompiler(context, cmdLine);

    cmdLine.addArg(filePath999);

    for (auto arg : input.testOptions->args)
    {
        cmdLine.addArg(arg);
    }

    // TODO: The compiler should probably define this automatically...
    cmdLine.addArg("-D");
    cmdLine.addArg("__SLANG__");

    cmdLine.addArg("-target");
    cmdLine.addArg(targetFormat);

    ExecuteResult exeRes;
    TEST_RETURN_ON_DONE(spawnAndWait(context, outputStem, input.spawnType, cmdLine, exeRes));

    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }

    // We ignore output to stdout, and only worry about what the compiler
    // wrote to stderr.

    ExecuteResult::ResultCode resultCode = exeRes.resultCode;

    String standardOutput = exeRes.standardOutput;
    String standardError = exeRes.standardError;

    // We construct a single output string that captures the results
    StringBuilder actualOutputBuilder;
    actualOutputBuilder.append("result code = ");
    actualOutputBuilder.append(resultCode);
    actualOutputBuilder.append("\nstandard error = {\n");
    actualOutputBuilder.append(standardError);
    actualOutputBuilder.append("}\nstandard output = {\n");
    actualOutputBuilder.append(standardOutput);
    actualOutputBuilder.append("}\n");

    String actualOutput = actualOutputBuilder.produceString();

    // Always fail if the compilation produced a failure, just
    // to catch situations where, e.g., command-line options parsing
    // caused the same error in both the Slang and fxc cases.
    return _validateOutput(context, input, actualOutput, resultCode != 0);
}

static TestResult runDXBCComparisonTest(TestContext* context, TestInput& input)
{
    return _runHLSLComparisonTest(context, input, "dxbc-assembly", "fxc");
}

static TestResult runDXILComparisonTest(TestContext* context, TestInput& input)
{
    return _runHLSLComparisonTest(context, input, "dxil-assembly", "dxc");
}

TestResult doGLSLComparisonTestRun(
    TestContext* context,
    TestInput& input,
    char const* langDefine,
    char const* passThrough,
    char const* outputKind,
    String* outOutput)
{
    auto filePath999 = input.filePath;
    auto outputStem = input.outputStem;

    CommandLine cmdLine;
    _initSlangCompiler(context, cmdLine);

    cmdLine.addArg(filePath999);

    if (langDefine)
    {
        cmdLine.addArg("-D");
        cmdLine.addArg(langDefine);
    }

    if (passThrough)
    {
        cmdLine.addArg("-pass-through");
        cmdLine.addArg(passThrough);
    }

    cmdLine.addArg("-target");
    cmdLine.addArg("spirv-assembly");

    for (auto arg : input.testOptions->args)
    {
        cmdLine.addArg(arg);
    }

    ExecuteResult exeRes;
    TEST_RETURN_ON_DONE(spawnAndWait(context, outputStem, input.spawnType, cmdLine, exeRes));

    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }

    ExecuteResult::ResultCode resultCode = exeRes.resultCode;

    String standardOuptut = exeRes.standardOutput;
    String standardError = exeRes.standardError;

    // We construct a single output string that captures the results
    StringBuilder outputBuilder;
    outputBuilder.append("result code = ");
    outputBuilder.append(resultCode);
    outputBuilder.append("\nstandard error = {\n");
    outputBuilder.append(standardError);
    outputBuilder.append("}\nstandard output = {\n");
    outputBuilder.append(standardOuptut);
    outputBuilder.append("}\n");

    String outputPath = outputStem + outputKind;
    String output = outputBuilder.produceString();

    *outOutput = output;

    return TestResult::Pass;
}

TestResult runGLSLComparisonTest(TestContext* context, TestInput& input)
{
    auto filePath999 = input.filePath;
    auto outputStem = input.outputStem;

    String expectedOutput;
    String actualOutput;

    TestResult hlslResult = doGLSLComparisonTestRun(
        context,
        input,
        "__GLSL__",
        "glslang",
        ".expected",
        &expectedOutput);
    TestResult slangResult =
        doGLSLComparisonTestRun(context, input, "__SLANG__", nullptr, ".actual", &actualOutput);

    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }

    // If either is ignored, the whole test is
    if (hlslResult == TestResult::Ignored || slangResult == TestResult::Ignored)
    {
        return TestResult::Ignored;
    }

    Slang::File::writeAllText(outputStem + ".expected", expectedOutput);
    Slang::File::writeAllText(outputStem + ".actual", actualOutput);

    if (hlslResult == TestResult::Fail)
        return TestResult::Fail;
    if (slangResult == TestResult::Fail)
        return TestResult::Fail;

    if (!StringUtil::areLinesEqual(
            actualOutput.getUnownedSlice(),
            expectedOutput.getUnownedSlice()))
    {
        context->getTestReporter()->dumpOutputDifference(expectedOutput, actualOutput);

        return TestResult::Fail;
    }

    return TestResult::Pass;
}

static void _addRenderTestOptions(const Options& options, CommandLine& ioCmdLine)
{
    if (!options.emitSPIRVDirectly)
    {
        ioCmdLine.addArg("-emit-spirv-via-glsl");
    }
}

static SlangResult _extractProfileTime(const UnownedStringSlice& text, double& timeOut)
{
    // Need to find the profile figure..
    LineParser parser(text);

    const auto lineStart = UnownedStringSlice::fromLiteral("profile-time=");
    for (auto line : parser)
    {
        if (line.startsWith(lineStart))
        {
            UnownedStringSlice remaining(line.begin() + lineStart.getLength(), line.end());
            remaining.trim();

            timeOut = stringToDouble(String(remaining));
            return SLANG_OK;
        }
    }

    return SLANG_FAIL;
}

TestResult runPerformanceProfile(TestContext* context, TestInput& input)
{
    auto outputStem = input.outputStem;

    CommandLine cmdLine;

    cmdLine.setExecutableLocation(ExecutableLocation(context->options.binDir, "render-test"));

    cmdLine.addArg(input.filePath);
    cmdLine.addArg("-performance-profile");

    _addRenderTestOptions(context->options, cmdLine);

    for (auto arg : input.testOptions->args)
    {
        cmdLine.addArg(arg);
    }

    ExecuteResult exeRes;
    TEST_RETURN_ON_DONE(spawnAndWait(context, outputStem, input.spawnType, cmdLine, exeRes));
    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }

    auto actualOutput = getOutput(exeRes);

    double time;
    if (SLANG_FAILED(_extractProfileTime(actualOutput.getUnownedSlice(), time)))
    {
        return TestResult::Fail;
    }

    context->getTestReporter()->addExecutionTime(time);

    return TestResult::Pass;
}


static double _textToDouble(const UnownedStringSlice& slice)
{
    Index size = Index(slice.getLength());
    // We have to zero terminate to be able to use atof
    const Index maxSize = 80;
    char buffer[maxSize + 1];

    size = (size > maxSize) ? maxSize : size;

    memcpy(buffer, slice.begin(), size);
    buffer[size] = 0;

    return atof(buffer);
}

static void _calcLines(const UnownedStringSlice& slice, List<UnownedStringSlice>& outLines)
{
    StringUtil::calcLines(slice, outLines);

    // Remove any trailing empty lines
    while (outLines.getCount())
    {
        if (outLines.getLast().trim() == UnownedStringSlice())
        {
            outLines.removeLast();
        }
        else
        {
            break;
        }
    }
}

static SlangResult _compareWithType(
    const UnownedStringSlice& actual,
    const UnownedStringSlice& ref,
    double differenceThreshold = 0.0001)
{
    typedef slang::TypeReflection::ScalarType ScalarType;

    ScalarType scalarType = ScalarType::None;

    // We just do straight comparison if there is no type

    List<UnownedStringSlice> linesActual, linesRef;

    _calcLines(actual, linesActual);
    _calcLines(ref, linesRef);

    // If there are more lines in actual, we just ignore them, to keep same behavior as before
    if (linesRef.getCount() < linesActual.getCount())
    {
        linesActual.setCount(linesRef.getCount());
    }

    if (linesActual.getCount() != linesRef.getCount())
    {
        return SLANG_FAIL;
    }

    for (Index i = 0; i < linesActual.getCount(); ++i)
    {
        const UnownedStringSlice lineActual = linesActual[i];
        const UnownedStringSlice lineRef = linesRef[i];

        if (lineActual.startsWith(UnownedStringSlice::fromLiteral("type:")))
        {
            if (lineActual != lineRef)
            {
                return SLANG_FAIL;
            }
            // Get the type
            List<UnownedStringSlice> split;
            StringUtil::split(lineActual, ':', split);

            if (split.getCount() != 2)
            {
                return SLANG_FAIL;
            }

            scalarType = TypeTextUtil::findScalarType(split[1].trim());
            continue;
        }

        switch (scalarType)
        {
        default:
            {
                if (lineActual.trim() != lineRef.trim())
                {
                    return SLANG_FAIL;
                }
                break;
            }
        case ScalarType::Float16:
        case ScalarType::Float32:
        case ScalarType::Float64:
            {

                // Compare as double
                double valueA = _textToDouble(lineActual);
                double valueB = _textToDouble(lineRef);

                if (!Math::AreNearlyEqual(valueA, valueB, differenceThreshold))
                {
                    return SLANG_FAIL;
                }
                break;
            }
        }
    }

    return SLANG_OK;
}

TestResult runComputeComparisonImpl(
    TestContext* context,
    TestInput& input,
    const char* const* langOpts,
    size_t numLangOpts)
{
    // TODO: delete any existing files at the output path(s) to avoid stale outputs leading to a
    // false pass
    auto filePath999 = input.filePath;
    auto outputStem = input.outputStem;

    CommandLine cmdLine;

    cmdLine.setExecutableLocation(ExecutableLocation(context->options.binDir, "render-test"));
    cmdLine.addArg(filePath999);

    _addRenderTestOptions(context->options, cmdLine);

    for (auto arg : input.testOptions->args)
    {
        cmdLine.addArg(arg);
    }

    for (int i = 0; i < int(numLangOpts); ++i)
    {
        cmdLine.addArg(langOpts[i]);
    }
    cmdLine.addArg("-o");
    auto actualOutputFile = outputStem + ".actual.txt";
    cmdLine.addArg(actualOutputFile);

#if _DEBUG
    // When using test server, any validation warning printed from the backend
    // gets misinterpreted as the result from the test.
    // This is due to the limitation that Slang RPC implementation expects only
    // one time communication.
    if (input.spawnType != SpawnType::UseTestServer)
        cmdLine.addArg("-enable-debug-layers");
#endif

    if (context->isExecuting())
    {
        // clear the stale actual output file first. This will allow us to detect error if
        // render-test fails and outputs nothing.
        File::writeAllText(actualOutputFile, "");
    }

    ExecuteResult exeRes;
    TEST_RETURN_ON_DONE(spawnAndWait(context, outputStem, input.spawnType, cmdLine, exeRes));

    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }

    // Check the stdout/stderr from the compiler process
    auto actualOutput = getOutput(exeRes);
    auto compileResult = _validateOutput(
        context,
        input,
        actualOutput,
        false,
        "result code = 0\nstandard error = {\n}\nstandard output = {\n}\n");

    // check against reference output
    String actualOutputContent;
    if (SLANG_FAILED(File::readAllText(actualOutputFile, actualOutputContent)))
    {
        context->getTestReporter()->messageFormat(
            TestMessageType::RunError,
            "Unable to read render-test output: %s\n",
            actualOutput.getBuffer());
        return TestResult::Fail;
    }

    String fileCheckPrefix;
    auto bufferResult =
        input.testOptions->getFileCheckBufferPrefix(fileCheckPrefix)
            ? _fileCheckTest(*context, input.filePath, fileCheckPrefix, actualOutputContent)
            : _fileComparisonTest(
                  *context,
                  input,
                  nullptr,
                  ".expected.txt",
                  actualOutputContent,
                  [](const auto& a, const auto& e) {
                      return SLANG_SUCCEEDED(
                          _compareWithType(a.getUnownedSlice(), e.getUnownedSlice()));
                  });
    return std::max(compileResult, bufferResult);
}

TestResult runSlangComputeComparisonTest(TestContext* context, TestInput& input)
{
    const char* langOpts[] = {"-slang", "-compute"};
    return runComputeComparisonImpl(context, input, langOpts, SLANG_COUNT_OF(langOpts));
}

TestResult runSlangComputeComparisonTestEx(TestContext* context, TestInput& input)
{
    return runComputeComparisonImpl(context, input, nullptr, 0);
}

TestResult runHLSLComputeTest(TestContext* context, TestInput& input)
{
    const char* langOpts[] = {"--hlsl-rewrite", "-compute"};
    return runComputeComparisonImpl(context, input, langOpts, SLANG_COUNT_OF(langOpts));
}

TestResult runSlangRenderComputeComparisonTest(TestContext* context, TestInput& input)
{
    const char* langOpts[] = {"-slang", "-gcompute"};
    return runComputeComparisonImpl(context, input, langOpts, SLANG_COUNT_OF(langOpts));
}

TestResult doRenderComparisonTestRun(
    TestContext* context,
    TestInput& input,
    char const* langOption,
    char const* outputKind,
    String* outOutput)
{
    // TODO: delete any existing files at the output path(s) to avoid stale outputs leading to a
    // false pass

    auto filePath = input.filePath;
    auto outputStem = input.outputStem;

    CommandLine cmdLine;

    cmdLine.setExecutableLocation(ExecutableLocation(context->options.binDir, "render-test"));
    cmdLine.addArg(filePath);

    _addRenderTestOptions(context->options, cmdLine);

    for (auto arg : input.testOptions->args)
    {
        cmdLine.addArg(arg);
    }

    cmdLine.addArg(langOption);
    cmdLine.addArg("-o");
    cmdLine.addArg(outputStem + outputKind + ".png");

    ExecuteResult exeRes;
    TEST_RETURN_ON_DONE(spawnAndWait(context, outputStem, input.spawnType, cmdLine, exeRes));

    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }

    ExecuteResult::ResultCode resultCode = exeRes.resultCode;

    String standardOutput = exeRes.standardOutput;
    String standardError = exeRes.standardError;

    // We construct a single output string that captures the results
    StringBuilder outputBuilder;
    outputBuilder.append("result code = ");
    outputBuilder.append(resultCode);
    outputBuilder.append("\nstandard error = {\n");
    outputBuilder.append(standardError);
    outputBuilder.append("}\nstandard output = {\n");
    outputBuilder.append(standardOutput);
    outputBuilder.append("}\n");

    String outputPath = outputStem + outputKind;
    String output = outputBuilder.produceString();

    *outOutput = output;

    // Always fail if the compilation produced a failure.
    if (exeRes.resultCode != 0)
    {
        return TestResult::Fail;
    }
    return TestResult::Pass;
}

class STBImage
{
public:
    typedef STBImage ThisType;

    /// Reset back to default initialized state (frees any image set)
    void reset();
    /// True if rhs has same size and amount of channels
    bool isComparable(const ThisType& rhs) const;

    /// The width in pixels
    int getWidth() const { return m_width; }
    /// The height in pixels
    int getHeight() const { return m_height; }
    /// The number of channels (typically held as bytes in order)
    int getNumChannels() const { return m_numChannels; }

    /// Get the contained pixels, nullptr if nothing loaded
    const unsigned char* getPixels() const { return m_pixels; }
    unsigned char* getPixels() { return m_pixels; }

    /// Read an image with filename. SLANG_OK on success
    SlangResult read(const char* filename);

    ~STBImage() { reset(); }

    int m_width = 0;
    int m_height = 0;
    int m_numChannels = 0;
    unsigned char* m_pixels = nullptr;
};

void STBImage::reset()
{
    if (m_pixels)
    {
        stbi_image_free(m_pixels);
        m_pixels = nullptr;
    }
    m_width = 0;
    m_height = 0;
    m_numChannels = 0;
}

SlangResult STBImage::read(const char* filename)
{
    reset();

    m_pixels = stbi_load(filename, &m_width, &m_height, &m_numChannels, 0);
    if (!m_pixels)
    {
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

bool STBImage::isComparable(const ThisType& rhs) const
{
    return (this == &rhs) || (m_width == rhs.m_width && m_height == rhs.m_height &&
                              m_numChannels == rhs.m_numChannels);
}


TestResult doImageComparison(TestContext* context, String const& filePath)
{
    auto reporter = context->getTestReporter();

    // Allow a difference in the low bits of the 8-bit result, just to play it safe
    static const int kAbsoluteDiffCutoff = 2;

    // Allow a relative 1% difference
    static const float kRelativeDiffCutoff = 0.01f;

    String expectedPath = filePath + ".expected.png";
    String actualPath = filePath + ".actual.png";

    STBImage expectedImage;
    if (SLANG_FAILED(expectedImage.read(expectedPath.getBuffer())))
    {
        reporter->messageFormat(
            TestMessageType::RunError,
            "Unable to load image ;%s'",
            expectedPath.getBuffer());
        return TestResult::Fail;
    }

    STBImage actualImage;
    if (SLANG_FAILED(actualImage.read(actualPath.getBuffer())))
    {
        reporter->messageFormat(
            TestMessageType::RunError,
            "Unable to load image ;%s'",
            actualPath.getBuffer());
        return TestResult::Fail;
    }

    if (!expectedImage.isComparable(actualImage))
    {
        reporter->messageFormat(
            TestMessageType::TestFailure,
            "Images are different sizes '%s' '%s'",
            actualPath.getBuffer(),
            expectedPath.getBuffer());
        return TestResult::Fail;
    }

    {
        const unsigned char* expectedPixels = expectedImage.getPixels();
        const unsigned char* actualPixels = actualImage.getPixels();

        const int height = actualImage.getHeight();
        const int width = actualImage.getWidth();
        const int numChannels = actualImage.getNumChannels();
        const int rowSize = width * numChannels;

        for (int y = 0; y < height; ++y)
        {
            for (int i = 0; i < rowSize; ++i)
            {
                int expectedVal = expectedPixels[i];
                int actualVal = actualPixels[i];

                int absoluteDiff = actualVal - expectedVal;
                if (absoluteDiff < 0)
                    absoluteDiff = -absoluteDiff;

                if (absoluteDiff < kAbsoluteDiffCutoff)
                {
                    // There might be a difference, but we'll consider it to be inside tolerance
                    continue;
                }

                float relativeDiff = 0.0f;
                if (expectedVal != 0)
                {
                    relativeDiff =
                        fabsf(float(actualVal) - float(expectedVal)) / float(expectedVal);

                    if (relativeDiff < kRelativeDiffCutoff)
                    {
                        // relative difference was small enough
                        continue;
                    }
                }

                // TODO: may need to do some local search sorts of things, to deal with
                // cases where vertex shader results lead to rendering that is off
                // by one pixel...

                const int x = i / numChannels;
                const int channelIndex = i % numChannels;

                reporter->messageFormat(
                    TestMessageType::TestFailure,
                    "image compare failure at (%d,%d) channel %d. expected %d got %d (absolute "
                    "error: %d, relative error: %f)\n",
                    x,
                    y,
                    channelIndex,
                    expectedVal,
                    actualVal,
                    absoluteDiff,
                    relativeDiff);

                // There was a difference we couldn't excuse!
                return TestResult::Fail;
            }

            expectedPixels += rowSize;
            actualPixels += rowSize;
        }
    }

    return TestResult::Pass;
}

TestResult runHLSLRenderComparisonTestImpl(
    TestContext* context,
    TestInput& input,
    char const* expectedArg,
    char const* actualArg)
{
    String _fileCheckPrefix;
    if (input.testOptions->getFileCheckPrefix(_fileCheckPrefix))
    {
        context->getTestReporter()->message(
            TestMessageType::RunError,
            "FileCheck testing isn't supported for HLSL render tests");
        return TestResult::Fail;
    }

    auto filePath = input.filePath;
    auto outputStem = input.outputStem;

    String expectedOutput;
    String actualOutput;

    // Run the expected test case only if we're not skipping reference image generation
    TestResult hlslResult = TestResult::Pass;
    if (!context->options.skipReferenceImageGeneration)
    {
        hlslResult =
            doRenderComparisonTestRun(context, input, expectedArg, ".expected", &expectedOutput);
        if (hlslResult != TestResult::Pass)
        {
            return hlslResult;
        }
    }

    // Always run the actual test case
    TestResult slangResult =
        doRenderComparisonTestRun(context, input, actualArg, ".actual", &actualOutput);
    if (slangResult != TestResult::Pass)
    {
        return slangResult;
    }

    if (context->isCollectingRequirements())
    {
        return TestResult::Pass;
    }

    // Save the expected output if we generated it
    if (!context->options.skipReferenceImageGeneration)
    {
        Slang::File::writeAllText(outputStem + ".expected", expectedOutput);
    }

    Slang::File::writeAllText(outputStem + ".actual", actualOutput);

    if (hlslResult == TestResult::Fail)
        return TestResult::Fail;
    if (slangResult == TestResult::Fail)
        return TestResult::Fail;

    // Compare text output only if we generated the expected output
    if (!context->options.skipReferenceImageGeneration && !StringUtil::areLinesEqual(
                                                              actualOutput.getUnownedSlice(),
                                                              expectedOutput.getUnownedSlice()))
    {
        context->getTestReporter()->dumpOutputDifference(expectedOutput, actualOutput);

        return TestResult::Fail;
    }

    // Next do an image comparison on the expected output images!

    TestResult imageCompareResult = doImageComparison(context, outputStem);
    if (imageCompareResult != TestResult::Pass)
        return imageCompareResult;

    return TestResult::Pass;
}

TestResult runHLSLRenderComparisonTest(TestContext* context, TestInput& input)
{
    return runHLSLRenderComparisonTestImpl(context, input, "-hlsl", "-slang");
}

TestResult runHLSLCrossCompileRenderComparisonTest(TestContext* context, TestInput& input)
{
    return runHLSLRenderComparisonTestImpl(context, input, "-slang", "-glsl-cross");
}

TestResult runHLSLAndGLSLRenderComparisonTest(TestContext* context, TestInput& input)
{
    return runHLSLRenderComparisonTestImpl(context, input, "-hlsl-rewrite", "-glsl-rewrite");
}

TestResult skipTest(TestContext* /* context */, TestInput& /*input*/)
{
    return TestResult::Ignored;
}

// based on command name, dispatch to an appropriate callback
struct TestCommandInfo
{
    char const* name;
    TestCallback callback;
    RenderApiFlags requiredRenderApiFlags; ///< An RenderApi types that are needed to run the tests
};

static const TestCommandInfo s_testCommandInfos[] = {
    {"SIMPLE", &runSimpleTest, 0},
    {"SIMPLE_EX", &runSimpleTest, 0},
    {"SIMPLE_LINE", &runSimpleLineTest, 0},
    {"INTERPRET", &runInterpreterTest, 0},
    {"REFLECTION", &runReflectionTest, 0},
    {"CPU_REFLECTION", &runReflectionTest, 0},
    {"COMMAND_LINE_SIMPLE", &runSimpleCompareCommandLineTest, 0},
    {"COMPARE_HLSL", &runDXBCComparisonTest, 0},
    {"COMPARE_DXIL", &runDXILComparisonTest, 0},
    {"COMPARE_HLSL_RENDER", &runHLSLRenderComparisonTest, 0},
    {"COMPARE_HLSL_CROSS_COMPILE_RENDER", &runHLSLCrossCompileRenderComparisonTest, 0},
    {"COMPARE_HLSL_GLSL_RENDER", &runHLSLAndGLSLRenderComparisonTest, 0},
    {"COMPARE_COMPUTE", &runSlangComputeComparisonTest, 0},
    {"COMPARE_COMPUTE_EX", &runSlangComputeComparisonTestEx, 0},
    {"HLSL_COMPUTE", &runHLSLComputeTest, 0},
    {"COMPARE_RENDER_COMPUTE", &runSlangRenderComputeComparisonTest, 0},
    {"COMPARE_GLSL", &runGLSLComparisonTest, 0},
    {"CROSS_COMPILE", &runCrossCompilerTest, 0},
    {"CPP_COMPILER_EXECUTE", &runCPPCompilerExecute, RenderApiFlag::CPU},
    {"CPP_COMPILER_SHARED_LIBRARY", &runCPPCompilerSharedLibrary, RenderApiFlag::CPU},
    {"CPP_COMPILER_COMPILE", &runCPPCompilerCompile, RenderApiFlag::CPU},
    {"PERFORMANCE_PROFILE", &runPerformanceProfile, 0},
    {"COMPILE", &runCompile, 0},
    {"DOC", &runDocTest, 0},
    {"LANG_SERVER", &runLanguageServerTest, 0},
    {"EXECUTABLE", &runExecutableTest, RenderApiFlag::CPU}};

const TestCommandInfo* _findTestCommandInfoByCommand(const UnownedStringSlice& name)
{
    for (const auto& command : s_testCommandInfos)
    {
        if (name == command.name)
        {
            return &command;
        }
    }
    return nullptr;
}

static RenderApiFlags _getRequiredRenderApisByCommand(const UnownedStringSlice& name)
{
    auto info = _findTestCommandInfoByCommand(name);
    return info ? info->requiredRenderApiFlags : 0;
}

TestResult runTest(
    TestContext* context,
    String const& filePath,
    String const& outputStem,
    String const& testName,
    TestOptions const& testOptions)
{
    // If we are collecting requirements and it's diagnostic test, we always run
    // (ie no requirements need to be captured - effectively it has 'no requirements')
    if (context->isCollectingRequirements() && testOptions.type == TestOptions::Diagnostic)
    {
        return TestResult::Pass;
    }

    auto testInfo = _findTestCommandInfoByCommand(testOptions.command.getUnownedSlice());

    if (testInfo)
    {
        TestInput testInput;
        testInput.filePath = filePath;
        testInput.outputStem = outputStem;
        testInput.testOptions = &testOptions;
        testInput.spawnType = context->options.defaultSpawnType;

        return testInfo->callback(context, testInput);
    }

    // No actual test runner found!
    return TestResult::Fail;
}

bool testCategoryMatches(TestCategory* sub, TestCategory* sup)
{
    auto ss = sub;
    while (ss)
    {
        if (ss == sup)
            return true;

        ss = ss->parent;
    }
    return false;
}

bool testCategoryMatches(
    TestCategory* categoryToMatch,
    const Dictionary<TestCategory*, TestCategory*>& categorySet)
{
    for (const auto& [_, category] : categorySet)
    {
        if (testCategoryMatches(categoryToMatch, category))
            return true;
    }
    return false;
}

bool testPassesCategoryMask(TestContext* context, TestOptions const& test)
{
    // Don't include a test we should filter out
    for (auto testCategory : test.categories)
    {
        if (testCategoryMatches(testCategory, context->options.excludeCategories))
            return false;
    }

    // Otherwise include any test the user asked for
    for (auto testCategory : test.categories)
    {
        if (testCategoryMatches(testCategory, context->options.includeCategories))
            return true;
    }

    // skip by default
    return false;
}

static void _calcSynthesizedTests(
    TestContext* context,
    RenderApiType synthRenderApiType,
    const List<TestDetails>& srcTests,
    List<TestDetails>& ioSynthTests)
{
    // Add the explicit parameter
    for (const auto& srcTest : srcTests)
    {
        const auto& requirements = srcTest.requirements;

        // Render tests use renderApis...
        // If it's an explicit test, we don't synth from it now

        // In the case of CUDA, we can only synth from a CPU source
        if (synthRenderApiType == RenderApiType::CUDA)
        {
            if (requirements.explicitRenderApi != RenderApiType::CPU)
            {
                continue;
            }

            // If the source language is defined, and it's

            const Index index = srcTest.options.args.indexOf("-source-language");
            if (index >= 0)
            {
                //
                const auto& language = srcTest.options.args[index + 1];
                SlangSourceLanguage sourceLanguage =
                    TypeTextUtil::findSourceLanguage(language.getUnownedSlice());

                bool isCrossCompile = true;

                switch (sourceLanguage)
                {
                case SLANG_SOURCE_LANGUAGE_GLSL:
                case SLANG_SOURCE_LANGUAGE_C:
                case SLANG_SOURCE_LANGUAGE_CPP:
                    {
                        isCrossCompile = false;
                    }
                default:
                    break;
                }

                if (!isCrossCompile)
                {
                    continue;
                }
            }
        }
        else
        {
            // TODO(JS): Arguably we should synthesize from explicit tests. In principal we can
            // remove the explicit api apply another although that may not always work. If it
            // doesn't use any render API or only uses CPU, we don't synthesize
            if (requirements.usedRenderApiFlags == 0 ||
                requirements.usedRenderApiFlags == RenderApiFlag::CPU ||
                requirements.explicitRenderApi != RenderApiType::Unknown)
            {
                continue;
            }
        }

        TestDetails synthTestDetails(srcTest.options);
        TestOptions& synthOptions = synthTestDetails.options;

        // If there's a category associated with this render api, add it to the synthesized test
        if (auto c = context->categorySet.find(RenderApiUtil::getApiName(synthRenderApiType)))
        {
            synthOptions.categories.add(c);
        }

        // Mark as synthesized
        synthOptions.isSynthesized = true;

        StringBuilder builder;
        builder << "-";
        builder << RenderApiUtil::getApiName(synthRenderApiType);

        synthOptions.args.add(builder);

        // If the target is vulkan remove the -hlsl option
        if (synthRenderApiType == RenderApiType::Vulkan)
        {
            const Index index = synthOptions.args.indexOf("-hlsl");
            if (index >= 0)
            {
                synthOptions.args.removeAt(index);
            }
        }
        else if (synthRenderApiType == RenderApiType::CUDA)
        {
            const Index index = synthOptions.args.indexOf("-cpu");
            if (index >= 0)
            {
                synthOptions.args.removeAt(index);
            }
        }

        // Work out the info about this tests
        context->setTestRequirements(&synthTestDetails.requirements);
        runTest(context, "", "", "", synthOptions);
        context->setTestRequirements(nullptr);

        // It does set the explicit render target
        SLANG_ASSERT(synthTestDetails.requirements.explicitRenderApi == synthRenderApiType);
        // Add to the tests
        ioSynthTests.add(synthTestDetails);
    }
}

static bool _canIgnore(TestContext* context, const TestDetails& details)
{
    if (details.options.isEnabled == false)
    {
        return true;
    }

    const auto& requirements = details.requirements;

    // Check if it's possible in principal to run this test with the render api flags used by this
    // test
    if (!context->canRunTestWithRenderApiFlags(requirements.usedRenderApiFlags))
    {
        return true;
    }

    // Are all the required backends available?
    if (((requirements.usedBackendFlags & context->availableBackendFlags) !=
         requirements.usedBackendFlags))
    {
        return true;
    }

    // If there are no render API requirements, then we don't need to ignore.
    if (requirements.usedRenderApiFlags == 0)
    {
        return false;
    }

    // Work out what render api flags are actually available, lazily
    const RenderApiFlags availableRenderApiFlags =
        requirements.usedRenderApiFlags ? _getAvailableRenderApiFlags(context) : 0;

    // Are all the required rendering apis available?
    if ((requirements.usedRenderApiFlags & availableRenderApiFlags) !=
        requirements.usedRenderApiFlags)
    {
        return true;
    }

    return false;
}

static SlangResult _runTestsOnFile(TestContext* context, String filePath)
{
    // Gather a list of tests to run
    FileTestList testList;

    SLANG_RETURN_ON_FAIL(_gatherTestsForFile(&context->categorySet, filePath, &testList));

    if (testList.tests.getCount() == 0)
    {
        // Test was explicitly ignored
        return SLANG_OK;
    }

    // Note cases where a test file exists, but we found nothing to run
    if (testList.tests.getCount() == 0)
    {
        context->getTestReporter()->addTest(filePath, TestResult::Ignored);
        return SLANG_OK;
    }

    RenderApiFlags apiUsedFlags = 0;
    RenderApiFlags explictUsedApiFlags = 0;

    {
        // We can get the test info for each of them
        for (auto& testDetails : testList.tests)
        {
            auto& requirements = testDetails.requirements;

            // Collect what the test needs (by setting restRequirements the test isn't actually run)
            context->setTestRequirements(&requirements);
            runTest(context, filePath, filePath, filePath, testDetails.options);

            //
            apiUsedFlags |= requirements.usedRenderApiFlags;
            explictUsedApiFlags |= (requirements.explicitRenderApi != RenderApiType::Unknown)
                                       ? (RenderApiFlags(1) << int(requirements.explicitRenderApi))
                                       : 0;
        }
        context->setTestRequirements(nullptr);
    }

    SLANG_ASSERT((apiUsedFlags & explictUsedApiFlags) == explictUsedApiFlags);

    const RenderApiFlags availableRenderApiFlags =
        apiUsedFlags ? _getAvailableRenderApiFlags(context) : 0;

    // If synthesized tests are wanted look into adding them
    if (context->options.synthesizedTestApis && availableRenderApiFlags)
    {
        List<TestDetails> synthesizedTests;

        // What render options do we want to synthesize
        RenderApiFlags missingApis =
            (~apiUsedFlags) & (context->options.synthesizedTestApis & availableRenderApiFlags);

        // const Index numInitialTests = testList.tests.getCount();

        while (missingApis)
        {
            const int index = ByteEncodeUtil::calcMsb8(missingApis);
            SLANG_ASSERT(index >= 0 && index <= int(RenderApiType::CountOf));

            const RenderApiType synthRenderApiType = RenderApiType(index);

            _calcSynthesizedTests(context, synthRenderApiType, testList.tests, synthesizedTests);

            // Disable the bit
            missingApis &= ~(RenderApiFlags(1) << index);
        }

        // Add all the synthesized tests
        testList.tests.addRange(synthesizedTests);
    }

    // We have found a test to run!
    int subTestCount = 0;
    for (auto& testDetails : testList.tests)
    {
        int subTestIndex = subTestCount++;

        // Check that the test passes our current category mask
        if (!testPassesCategoryMask(context, testDetails.options))
        {
            continue;
        }

        // Work out the test stem

        StringBuilder outputStem;
        outputStem << filePath;
        if (subTestIndex != 0)
        {
            outputStem << "." << subTestIndex;
        }

        // Work out the test name - taking into account render api / if synthesized
        StringBuilder testName(outputStem);

        if (testDetails.options.isSynthesized)
        {
            testName << " syn";
        }

        const auto& requirements = testDetails.requirements;

        // Display list of used apis on render test
        if (requirements.usedRenderApiFlags)
        {
            RenderApiFlags usedFlags = requirements.usedRenderApiFlags;
            testName << " (";
            bool isPrev = false;
            while (usedFlags)
            {
                const int index = ByteEncodeUtil::calcMsb8(usedFlags);
                const RenderApiType renderApiType = RenderApiType(index);
                if (isPrev)
                {
                    testName << ",";
                }
                testName << RenderApiUtil::getApiName(renderApiType);

                // Disable bit
                usedFlags &= ~(RenderApiFlags(1) << index);
                isPrev = true;
            }
            testName << ")";
        }

        // Report the test and run/ignore
        {
            TestReporter::TestScope scope(context->getTestReporter(), testName);

            TestResult testResult = TestResult::Fail;

            // If this test can be ignored
            if (_canIgnore(context, testDetails))
            {
                testResult = TestResult::Ignored;
                context->getTestReporter()->addResult(testResult);
            }
            else
            {
                testResult = runTest(context, filePath, outputStem, testName, testDetails.options);
                if (testResult == TestResult::Fail &&
                    !context->getTestReporter()->m_expectedFailureList.contains(testName))
                {
                    RefPtr<FileTestInfoImpl> fileTestInfo = new FileTestInfoImpl();
                    fileTestInfo->filePath = filePath;
                    fileTestInfo->testName = testName;
                    fileTestInfo->outputStem = outputStem;
                    fileTestInfo->options = testDetails.options;

                    std::lock_guard lock(context->mutexFailedTests);
                    context->failedFileTests.add(fileTestInfo);
                }
                else
                {
                    context->getTestReporter()->addResult(testResult);
                }
            }


            // Could determine if to continue or not here... based on result
        }
    }

    return SLANG_OK;
}


static bool endsWithAllowedExtension(TestContext* /*context*/, String filePath)
{
    char const* allowedExtensions[] = {
        ".slang",
        ".hlsl",
        ".fx",
        ".glsl",
        ".vert",
        ".frag",
        ".geom",
        ".tesc",
        ".tese",
        ".comp",
        ".internal",
        ".ahit",
        ".chit",
        ".miss",
        ".rgen",
        ".c",
        ".cpp",
        ".cu",
    };

    for (auto allowedExtension : allowedExtensions)
    {
        if (filePath.endsWith(allowedExtension))
            return true;
    }

    return false;
}

static bool shouldRunTest(TestContext* context, String filePath)
{
    if (!endsWithAllowedExtension(context, filePath))
        return false;

    if (!context->options.testPrefixes.getCount())
    {
        return true;
    }

    // If we have prefixes, it has to match one of them
    for (auto& p : context->options.testPrefixes)
    {
        if (filePath.startsWith(p))
        {
            return true;
        }
    }
    return false;
}

void getFilesInDirectory(String directoryPath, List<String>& files)
{
    {
        List<String> localFiles;
        DirectoryUtil::findFiles(directoryPath, localFiles);
        files.addRange(localFiles);
    }
    {
        List<String> subDirs;
        DirectoryUtil::findDirectories(directoryPath, subDirs);
        for (auto subDir : subDirs)
        {
            getFilesInDirectory(subDir, files);
        }
    }
}

template<typename F>
void runTestsInParallel(TestContext* context, int count, const F& f)
{
    auto originalReporter = context->getTestReporter();
    std::atomic<int> consumePtr;
    consumePtr = 0;
    auto threadFunc = [&](int threadId)
    {
        TestReporter reporter;
        reporter.init(context->options.outputMode, context->options.expectedFailureList, true);
        TestReporter::SuiteScope suiteScope(&reporter, "tests");
        context->setThreadIndex(threadId);
        context->setTestReporter(&reporter);
        do
        {
            int index = consumePtr.fetch_add(1);
            if (index >= count)
                break;
            f(index);
        } while (true);
        {
            std::lock_guard<std::mutex> lock(context->mutex);
            originalReporter->consolidateWith(&reporter);
        }
        context->setTestReporter(nullptr);
    };
    List<std::thread> threads;
    for (int threadId = 0; threadId < context->options.serverCount; threadId++)
    {
        threads.add(std::thread(threadFunc, threadId));
    }
    for (auto& t : threads)
        t.join();
    context->setTestReporter(originalReporter);
}

void runTestsInDirectory(TestContext* context)
{
    List<String> files;
    getFilesInDirectory(context->options.testDir, files);
    auto processFile = [&](String file)
    {
        if (shouldRunTest(context, file))
        {
            //            fprintf(stderr, "slang-test: found '%s'\n", file.getBuffer());
            if (SLANG_FAILED(_runTestsOnFile(context, file)))
            {
                {
                    TestReporter::TestScope scope(context->getTestReporter(), file);
                    context->getTestReporter()->message(
                        TestMessageType::RunError,
                        "slang-test: unable to parse test");

                    context->getTestReporter()->addResult(TestResult::Fail);
                }

                // Output there was some kind of error trying to run the tests on this file
                // fprintf(stderr, "slang-test: unable to parse test '%s'\n", file.getBuffer());
            }
        }
    };
    bool useMultiThread = false;
    switch (context->options.defaultSpawnType)
    {
    case SpawnType::UseFullyIsolatedTestServer:
    case SpawnType::UseTestServer:
        useMultiThread = true;
        break;
    }
    if (context->options.serverCount == 1)
    {
        useMultiThread = false;
    }
    if (!useMultiThread)
    {
        for (auto file : files)
        {
            processFile(file);
        }
    }
    else
    {
        runTestsInParallel(
            context,
            (int)files.getCount(),
            [&](int index) { processFile(files[index]); });
    }
}

static void _disableCPPBackends(TestContext* context)
{
    const SlangPassThrough cppPassThrus[] = {
        SLANG_PASS_THROUGH_GENERIC_C_CPP,
        SLANG_PASS_THROUGH_VISUAL_STUDIO,
        SLANG_PASS_THROUGH_CLANG,
        SLANG_PASS_THROUGH_GCC,
    };

    for (auto passThru : cppPassThrus)
    {
        context->availableBackendFlags &= ~(PassThroughFlags(1) << int(passThru));
        context->availableRenderApiFlags &= ~(RenderApiFlag::CPU);
        context->options.enabledApis &= ~(RenderApiFlag::CPU);
    }
}

static void _disableD3D12Backend(TestContext* context)
{
    context->options.enabledApis &= ~(RenderApiFlag::D3D12);
}

static TestResult _asTestResult(ToolReturnCode retCode)
{
    switch (retCode)
    {
    default:
        return TestResult::Fail;
    case ToolReturnCode::Success:
        return TestResult::Pass;
    case ToolReturnCode::Ignored:
        return TestResult::Ignored;
    }
}

/// Loads a DLL containing unit test functions and run them one by one.
static SlangResult runUnitTestModule(
    TestContext* context,
    TestOptions& testOptions,
    SpawnType spawnType,
    const char* moduleName)
{
    ISlangSharedLibraryLoader* loader = DefaultSharedLibraryLoader::getSingleton();
    ComPtr<ISlangSharedLibrary> moduleLibrary;

    SLANG_RETURN_ON_FAIL(loader->loadSharedLibrary(
        Path::combine(context->dllDirectoryPath, moduleName).getBuffer(),
        moduleLibrary.writeRef()));

    UnitTestGetModuleFunc getModuleFunc =
        (UnitTestGetModuleFunc)moduleLibrary->findFuncByName("slangUnitTestGetModule");
    if (!getModuleFunc)
        return SLANG_FAIL;

    IUnitTestModule* testModule = getModuleFunc();
    if (!testModule)
        return SLANG_FAIL;

    UnitTestContext unitTestContext;
    unitTestContext.slangGlobalSession = context->getSession();
    unitTestContext.workDirectory = "";
    unitTestContext.enabledApis = context->options.enabledApis;
    unitTestContext.executableDirectory = context->exeDirectoryPath.getBuffer();

    auto testCount = testModule->getTestCount();

    struct TestItem
    {
        UnitTestFunc testFunc;
        String testName;
        String command;
    };

    List<TestItem> tests;

    // Discover all tests first.
    for (SlangInt i = 0; i < testCount; i++)
    {
        auto testFunc = testModule->getTestFunc(i);
        auto testName = testModule->getTestName(i);

        StringBuilder filePath;
        filePath << moduleName << "/" << testName << ".internal";
        auto command = filePath.produceString();

        if (shouldRunTest(context, command))
        {
            if (testPassesCategoryMask(context, testOptions))
            {
                tests.add(TestItem{testFunc, testName, command});
            }
        }
    }

    auto runUnitTest = [&](TestItem test)
    {
        auto reporter = context->getTestReporter();
        TestOptions options = testOptions;
        options.command = test.command;

        if (spawnType == SpawnType::UseTestServer ||
            spawnType == SpawnType::UseFullyIsolatedTestServer)
        {
            TestServerProtocol::ExecuteUnitTestArgs args;
            args.enabledApis = context->options.enabledApis;
            args.moduleName = moduleName;
            args.testName = test.testName;

            {
                TestReporter::TestScope scopeTest(reporter, options.command);
                ExecuteResult exeRes;

                SlangResult rpcRes = _executeRPC(
                    context,
                    spawnType,
                    TestServerProtocol::ExecuteUnitTestArgs::g_methodName,
                    &args,
                    exeRes);
                auto testResult = _asTestResult(ToolReturnCode(exeRes.resultCode));

                bool isFailed = (SLANG_FAILED(rpcRes) || testResult == TestResult::Fail);

                // If the test fails, output any output - which might give information about
                // individual tests that have failed.
                if (isFailed)
                {
                    String output = getOutput(exeRes);
                    reporter->message(TestMessageType::TestFailure, output.getBuffer());
                }

                if (isFailed && !context->isRetry &&
                    !context->getTestReporter()->m_expectedFailureList.contains(test.testName))
                {
                    std::lock_guard lock(context->mutexFailedTests);
                    context->failedUnitTests.add(test.command);
                }
                else
                {
                    reporter->addResult(testResult);
                }
            }
        }
        else
        {
            TestReporter::TestScope scopeTest(reporter, options.command);

            // TODO(JS): Problem here could be exception not handled properly across
            // shared library boundary.
            testModule->setTestReporter(reporter);
            try
            {
                test.testFunc(&unitTestContext);
            }
            catch (...)
            {
                reporter->message(
                    TestMessageType::TestFailure,
                    "Exception was thrown during execution");
                reporter->addResult(TestResult::Fail);
            }
        }
    };

    bool useMultiThread = false;
    if (spawnType == SpawnType::UseTestServer || spawnType == SpawnType::UseFullyIsolatedTestServer)
    {
        if (context->options.serverCount > 1)
        {
            useMultiThread = true;
        }
    }

    if (useMultiThread)
    {
        runTestsInParallel(
            context,
            (int)tests.getCount(),
            [&](int index) { runUnitTest(tests[index]); });
    }
    else
    {
        auto reporter = TestReporter::get();

        testModule->setTestReporter(reporter);

        for (auto t : tests)
            runUnitTest(t);
    }

    testModule->destroy();
    return SLANG_OK;
}

SlangResult innerMain(int argc, char** argv)
{
    auto stdWriters = StdWriters::initDefaultSingleton();

    // The context holds useful things used during testing
    TestContext context;
    SLANG_RETURN_ON_FAIL(SLANG_FAILED(context.init(argv[0])))

    auto& categorySet = context.categorySet;

    // Set up our test categories here
    auto fullTestCategory = categorySet.add("full", nullptr);
    auto quickTestCategory = categorySet.add("quick", fullTestCategory);
    auto smokeTestCategory = categorySet.add("smoke", quickTestCategory);
    auto renderTestCategory = categorySet.add("render", fullTestCategory);
    /*auto computeTestCategory = */ categorySet.add("compute", fullTestCategory);
    auto vulkanTestCategory = categorySet.add("vulkan", fullTestCategory);
    auto unitTestCategory = categorySet.add("unit-test", fullTestCategory);
    auto cudaTestCategory = categorySet.add("cuda", fullTestCategory);
    auto optixTestCategory = categorySet.add("optix", cudaTestCategory);

    auto waveTestCategory = categorySet.add("wave", fullTestCategory);
    auto waveMaskCategory = categorySet.add("wave-mask", waveTestCategory);
    auto waveActiveCategory = categorySet.add("wave-active", waveTestCategory);

    auto compatibilityIssueCategory = categorySet.add("compatibility-issue", fullTestCategory);

    auto sharedLibraryCategory = categorySet.add("shared-library", fullTestCategory);

#if SLANG_WINDOWS_FAMILY
    auto windowsCategory = categorySet.add("windows", fullTestCategory);
#endif

#if SLANG_UNIX_FAMILY
    auto unixCategory = categorySet.add("unix", fullTestCategory);
#endif

#if SLANG_PTR_IS_64
    auto ptr64Category = categorySet.add("64-bit", fullTestCategory);
#else
    auto ptr32Category = categorySet.add("32-bit", fullTestCategory);
#endif

    // An un-categorized test will always belong to the `full` category
    categorySet.defaultCategory = fullTestCategory;

    // All following values are initialized to '0', so null.
    TestCategory* passThroughCategories[SLANG_PASS_THROUGH_COUNT_OF] = {nullptr};

    // Work out what backends/pass-thrus are available
    {
        SlangSession* session = context.getSession();

        auto out = StdWriters::getOut();
        out.print("Supported backends:");

        for (int i = 0; i < SLANG_PASS_THROUGH_COUNT_OF; ++i)
        {
            const SlangPassThrough passThru = SlangPassThrough(i);
            if (passThru == SLANG_PASS_THROUGH_NONE)
            {
                continue;
            }

            if (SLANG_SUCCEEDED(session->checkPassThroughSupport(passThru)))
            {
                context.availableBackendFlags |= PassThroughFlags(1) << int(i);

                StringBuilder buf;

                auto name = TypeTextUtil::getPassThroughName(passThru);

                buf << " " << name;

                SLANG_ASSERT(passThroughCategories[i] == nullptr);
                passThroughCategories[i] = categorySet.add(buf.getBuffer() + 1, fullTestCategory);

                out.write(buf.getBuffer(), buf.getLength());
            }
        }

        out.print("\n");
    }

    {
        SlangSession* session = context.getSession();

        const bool hasLlvm =
            SLANG_SUCCEEDED(session->checkPassThroughSupport(SLANG_PASS_THROUGH_LLVM));
        const auto hostCallableCompiler = session->getDownstreamCompilerForTransition(
            SLANG_CPP_SOURCE,
            SLANG_SHADER_HOST_CALLABLE);

        if (hasLlvm && hostCallableCompiler == SLANG_PASS_THROUGH_LLVM && SLANG_PROCESSOR_X86)
        {
            // TODO(JS)
            // For some reason host-callable with llvm/double produces different results on x86
        }
        else
        {
            // Special category to mark a test only works for targets that work correctly with
            // double (ie not x86/llvm)
            categorySet.add("war-double-host-callable", fullTestCategory);
        }
    }

    // Working out what renderApis is worked on on demand through
    // _getAvailableRenderApiFlags()

    {
        // We can set the slangc command line tool, to just use the function defined here
        context.setInnerMainFunc("slangc", &SlangCTool::innerMain);
    }

    {
        // We can set the slangc command line tool, to just use the function defined here
        context.setInnerMainFunc("slangi", &SlangITool::innerMain);
    }

    SLANG_RETURN_ON_FAIL(
        Options::parse(argc, argv, &categorySet, StdWriters::getError(), &context.options));

    Options& options = context.options;

    context.setMaxTestRunnerThreadCount(options.serverCount);

    // Set up the prelude/s
    TestToolUtil::setSessionDefaultPreludeFromExePath(argv[0], context.getSession());

    if (options.outputMode == TestOutputMode::TeamCity)
    {
        // On TeamCity CI there is an issue with unix/linux targets where test system may be
        // different from the build system That we rely on having compilation tools present such
        // that on x64 systems we can build x86 binaries, and that appears to not always be the
        // case. For now we only allow CPP backends to run on x86_64 targets
#if SLANG_UNIX_FAMILY && !SLANG_PROCESSOR_X86_64
        _disableCPPBackends(&context);
#endif
    }

#if SLANG_PROCESSOR_X86
    // Disable d3d12 tests on x86 right now since dxc for 32-bit windows doesn't seem to recognize
    // sm_6_6.
    _disableD3D12Backend(&context);
#endif

    if (options.subCommand.getLength())
    {
        // Get the function from the tool
        auto func = context.getInnerMainFunc(options.binDir, options.subCommand);
        if (!func)
        {
            StdWriters::getError().print(
                "error: Unable to launch tool '%s'\n",
                options.subCommand.getBuffer());
            return SLANG_FAIL;
        }

        // Copy args to a char* list
        const auto& srcArgs = options.subCommandArgs;
        List<const char*> args;
        args.setCount(srcArgs.getCount());
        for (Index i = 0; i < srcArgs.getCount(); ++i)
        {
            args[i] = srcArgs[i].getBuffer();
        }

        return func(
            StdWriters::getSingleton(),
            context.getSession(),
            int(args.getCount()),
            args.getBuffer());
    }

    if (options.includeCategories.getCount() == 0)
    {
        options.includeCategories.add(fullTestCategory, fullTestCategory);
    }

    // Don't include OptiX tests unless the client has explicit opted into them.
    if (!options.includeCategories.containsKey(optixTestCategory))
    {
        options.excludeCategories.add(optixTestCategory, optixTestCategory);
    }

    // Exclude rendering tests when building under AppVeyor.
    //
    // TODO: this is very ad hoc, and we should do something cleaner.
    if (options.outputMode == TestOutputMode::AppVeyor)
    {
        options.excludeCategories.add(renderTestCategory, renderTestCategory);
        options.excludeCategories.add(vulkanTestCategory, vulkanTestCategory);
    }

    {
        // Setup the reporter
        TestReporter reporter;
        SLANG_RETURN_ON_FAIL(reporter.init(options.outputMode, options.expectedFailureList));

        context.setTestReporter(&reporter);

        reporter.m_dumpOutputOnFailure = options.dumpOutputOnFailure;
        reporter.m_isVerbose = options.shouldBeVerbose;
        reporter.m_hideIgnored = options.hideIgnored;

        {
            TestReporter::SuiteScope suiteScope(&reporter, "tests");
            // Enumerate test files according to policy
            runTestsInDirectory(&context);
        }

        // Run the unit tests (these are internal C++ tests - not specified via files in a
        // directory) They are registered with SLANG_UNIT_TEST macro
        //
        //
        if (context.canRunUnitTests())
        {
            TestReporter::SuiteScope suiteScope(&reporter, "unit tests");
            TestReporter::set(&reporter);

            for (bool isRetry : {false, true})
            {
                auto spawnType = context.getFinalSpawnType();

                context.isRetry = false;
                if (isRetry)
                {
                    if (context.failedUnitTests.getCount() == 0)
                        break;

                    printf("Retrying unit tests...\n");
                    context.isRetry = true;
                    context.options.testPrefixes = context.failedUnitTests;
                    context.failedUnitTests.clear();
                    spawnType = SpawnType::Default;
                }

                // Run the unit tests
                {
                    TestOptions testOptions;
                    testOptions.categories.add(unitTestCategory);
                    testOptions.categories.add(smokeTestCategory);
                    runUnitTestModule(&context, testOptions, spawnType, "slang-unit-test-tool");
                }

                {
                    TestOptions testOptions;
                    testOptions.categories.add(unitTestCategory);
                    runUnitTestModule(&context, testOptions, spawnType, "gfx-unit-test-tool");
                }
            }

            TestReporter::set(nullptr);
        }

        // If we have a couple failed tests, they maybe intermittent failures due to parallel
        // excution or driver instability. We can try running them again.
        static constexpr int kFailedTestLimitForRetry = 16;
        if (context.failedFileTests.getCount() <= kFailedTestLimitForRetry)
        {
            if (context.failedFileTests.getCount() > 0)
                printf("Retrying %d failed tests...\n", (int)context.failedFileTests.getCount());
            for (auto& test : context.failedFileTests)
            {
                FileTestInfoImpl* fileTestInfo = static_cast<FileTestInfoImpl*>(test.Ptr());
                TestReporter::SuiteScope suiteScope(&reporter, "tests");
                TestReporter::TestScope scope(&reporter, fileTestInfo->testName);
                auto newResult = runTest(
                    &context,
                    fileTestInfo->filePath,
                    fileTestInfo->outputStem,
                    fileTestInfo->testName,
                    fileTestInfo->options);
                reporter.addResult(newResult);
            }
        }
        else
        {
            // If there are too many failed tests, don't bother retrying.
            for (auto& test : context.failedFileTests)
            {
                FileTestInfoImpl* fileTestInfo = static_cast<FileTestInfoImpl*>(test.Ptr());
                TestReporter::SuiteScope suiteScope(&reporter, "tests");
                TestReporter::TestScope scope(&reporter, fileTestInfo->testName);
                reporter.addResult(TestResult::Fail);
            }
        }

        reporter.outputSummary();
        return reporter.didAllSucceed() ? SLANG_OK : SLANG_FAIL;
    }
}

int main(int argc, char** argv)
{
    const SlangResult res = innerMain(argc, argv);
    slang::shutdown();
    Slang::RttiInfo::deallocateAll();

#ifdef _MSC_VER
    _CrtDumpMemoryLeaks();
#endif
    return SLANG_SUCCEEDED(res) ? 0 : 1;
}
