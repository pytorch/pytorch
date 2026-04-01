// unit-test-record-replay.cpp

#include "../../source/core/slang-http.h"
#include "../../source/core/slang-io.h"
#include "../../source/core/slang-process-util.h"
#include "../../source/core/slang-random-generator.h"
#include "../../source/core/slang-string-util.h"
#include "unit-test/slang-unit-test.h"

#include <chrono>
#include <thread>

using namespace Slang;

static SlangResult createProcess(
    UnitTestContext* context,
    const char* processName,
    const List<String>* optArgs,
    RefPtr<Process>& outProcess)
{
    CommandLine cmdLine;
    cmdLine.setExecutableLocation(ExecutableLocation(context->executableDirectory, processName));
    if (optArgs)
    {
        cmdLine.m_args.addRange(optArgs->getBuffer(), optArgs->getCount());
    }

    SLANG_RETURN_ON_FAIL(Process::create(cmdLine, Process::Flag::AttachDebugger, outProcess));

    return SLANG_OK;
}

struct entryHashInfo
{
    int64_t callIdx = -1;
    int64_t targetIndex = -1;
    int64_t entryPointIndex = -1;
    String hash;
};

static SlangResult parseHashes(List<String> const& lines, List<entryHashInfo>& outHashes)
{
    SlangResult res = SLANG_OK;

    for (const auto& line : lines)
    {
        List<UnownedStringSlice> tokens;
        Index skipCharacters = line.indexOf(UnownedStringSlice("[slang-record-replay]:"));
        if (skipCharacters == -1)
        {
            skipCharacters = 0;
        }
        else
        {
            skipCharacters += strlen("[slang-record-replay]:");
        }
        StringUtil::split(UnownedStringSlice(line.getBuffer() + skipCharacters), ',', tokens);

        if (tokens.getCount() != 4)
        {
            return SLANG_FAIL;
        }

        entryHashInfo hashInfo;
        auto extractToken = [](const UnownedStringSlice& token,
                               const char splitChar,
                               UnownedStringSlice& outToken) -> SlangResult
        {
            List<UnownedStringSlice> subTokens;
            StringUtil::split(token, splitChar, subTokens);
            if (subTokens.getCount() != 2)
            {
                return SLANG_FAIL;
            }
            outToken = subTokens[1];
            return SLANG_OK;
        };

        {
            UnownedStringSlice subToken;
            SLANG_RETURN_ON_FAIL(extractToken(tokens[0], ':', subToken));
            int64_t outNumer = 0;
            StringUtil::parseInt64(subToken, outNumer);
            hashInfo.callIdx = outNumer;
        }

        {
            UnownedStringSlice subToken;
            SLANG_RETURN_ON_FAIL(extractToken(tokens[1], ':', subToken));
            int64_t outNumer = 0;
            StringUtil::parseInt64(subToken, outNumer);
            hashInfo.entryPointIndex = outNumer;
        }

        {
            UnownedStringSlice subToken;
            SLANG_RETURN_ON_FAIL(extractToken(tokens[2], ':', subToken));
            int64_t outNumer = 0;
            StringUtil::parseInt64(subToken, outNumer);
            hashInfo.targetIndex = outNumer;
        }

        {
            UnownedStringSlice subToken;
            SLANG_RETURN_ON_FAIL(extractToken(tokens[3], ':', subToken));
            // remove the white space after ":"
            hashInfo.hash = subToken.begin() + 1;
        }

        outHashes.add(hashInfo);
    }
    return res;
}

static int writeEnvironmentVariable(const char* key, const char* val)
{
#ifdef _WIN32
    String var = String(key) + "=" + val;
    return _putenv(var.getBuffer());
#else
    return setenv(key, val, 1);
#endif
}

static bool enableRecordLayer()
{
    int retCode = writeEnvironmentVariable("SLANG_RECORD_LAYER", "1");
    return retCode == 0;
}

static bool disableRecordLayer()
{
    int retCode = writeEnvironmentVariable("SLANG_RECORD_LAYER", "0");
    return retCode == 0;
}

static bool enableLogInReplayer()
{
    int retCode = writeEnvironmentVariable("SLANG_RECORD_LOG_LEVEL", "3");
    return retCode == 0;
}

static bool disableLogInReplayer()
{
    int retCode = writeEnvironmentVariable("SLANG_RECORD_LOG_LEVEL", "0");
    return retCode == 0;
}

static void findRecordFileName(List<String>* fileNames)
{
    struct Visitor : Path::Visitor
    {
        void accept(Path::Type type, const UnownedStringSlice& filename) SLANG_OVERRIDE
        {
            if (type == Path::Type::File)
            {
                m_fileNames->add(filename);
            }
        }
        Visitor(List<String>* fileNames)
            : m_fileNames(fileNames)
        {
        }
        List<String>* m_fileNames;
    };

    Visitor visitor(fileNames);
    Path::find("slang-record", "*.cap", &visitor);
}

static SlangResult launchProcessAndReadStdout(
    UnitTestContext* context,
    const List<String>& optArgs,
    const char* exampleName,
    RefPtr<Process>& process,
    ExecuteResult& exeRes)
{
    StringBuilder msgBuilder;
    SlangResult res = createProcess(context, exampleName, &optArgs, process);
    if (SLANG_FAILED(res))
    {
        msgBuilder << "Failed to launch process of '" << exampleName << "'\n";
        getTestReporter()->message(TestMessageType::TestFailure, msgBuilder.toString().getBuffer());
        return res;
    }

    res = ProcessUtil::readUntilTermination(process, exeRes);
    if (SLANG_FAILED(res))
    {
        msgBuilder << "Failed to read stdout from '" << exampleName << "'\n";
        msgBuilder << "process ret code: " << exeRes.resultCode;
        getTestReporter()->message(TestMessageType::TestFailure, msgBuilder.toString().getBuffer());
        return res;
    }

    if (exeRes.resultCode != 0)
    {
        msgBuilder << "'" << exampleName << "' exits with failure\n";
        msgBuilder << "Process ret code: " << exeRes.resultCode << "\n";
        msgBuilder << "Standard output:\n" << exeRes.standardOutput;
        msgBuilder << "Standard error:\n" << exeRes.standardError;
        getTestReporter()->message(TestMessageType::TestFailure, msgBuilder.toString().getBuffer());
        return SLANG_FAIL;
    }

    if (exeRes.standardOutput.getLength() == 0)
    {
        msgBuilder << "No stdout found in '" << exampleName << "'\n";
        msgBuilder << "Standard error: " << exeRes.standardError;
        getTestReporter()->message(TestMessageType::TestFailure, msgBuilder.toString().getBuffer());
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

static SlangResult runExample(
    UnitTestContext* context,
    const char* exampleName,
    List<entryHashInfo>& outHashes)
{
    SlangResult finalRes = SLANG_OK;

    RefPtr<Process> process;
    ExecuteResult exeRes;
    List<String> optArgs;
    optArgs.add("--test-mode");

    StringBuilder msgBuilder;
    SlangResult res = SLANG_OK;

    enableRecordLayer();
    res = launchProcessAndReadStdout(context, optArgs, exampleName, process, exeRes);
    disableRecordLayer();

    if (SLANG_FAILED(res))
    {
        return res;
    }

    List<String> hashLines;
    for (auto line : LineParser(exeRes.standardOutput.getUnownedSlice()))
    {
        if (line.getLength() == 0)
        {
            continue;
        }

        if (line.indexOf(UnownedStringSlice("hash:")) == -1)
        {
            continue;
        }

        hashLines.add(line);
    }

    res = parseHashes(hashLines, outHashes);
    if (SLANG_FAILED(res))
    {
        msgBuilder << "Failed to parse hash from stdout of '" << exampleName << "'\n";
        getTestReporter()->message(TestMessageType::TestFailure, msgBuilder.toString().getBuffer());
        return res;
    }

    return SLANG_OK;
}

static SlangResult replayExample(UnitTestContext* context, List<entryHashInfo>& outHashes)
{
    List<String> fileNames;
    findRecordFileName(&fileNames);
    if (fileNames.getCount() == 0)
    {
        getTestReporter()->message(TestMessageType::TestFailure, "No record files found\n");
        return SLANG_FAIL;
    }

    List<String> optArgs;
    String recordFileName = Path::combine("slang-record", fileNames[0]);
    optArgs.add(recordFileName.getBuffer());

    RefPtr<Process> process;
    ExecuteResult exeRes;

    StringBuilder msgBuilder;
    msgBuilder << "replay the test\n";

    enableLogInReplayer();
    SlangResult res = launchProcessAndReadStdout(context, optArgs, "slang-replay", process, exeRes);
    disableLogInReplayer();

    if (SLANG_FAILED(res))
    {
        return res;
    }

    List<String> hashLines;
    for (auto line : LineParser(exeRes.standardOutput.getUnownedSlice()))
    {
        if (line.getLength() == 0)
        {
            continue;
        }

        if (line.indexOf(UnownedStringSlice("hash:")) == -1)
        {
            continue;
        }

        hashLines.add(line);
    }

    res = parseHashes(hashLines, outHashes);
    if (SLANG_FAILED(res))
    {
        msgBuilder << "Failed to parse hash from stdout of 'slang-replay'\n";
        getTestReporter()->message(TestMessageType::TestFailure, msgBuilder.toString().getBuffer());
        return SLANG_FAIL;
    }

    return SLANG_OK;
}

static SlangResult resultCompare(
    List<entryHashInfo> const& expectHashes,
    List<entryHashInfo> const& resultHashes)
{
    if (expectHashes.getCount() == 0)
    {
        getTestReporter()->message(TestMessageType::TestFailure, "No hash found\n");
        return SLANG_FAIL;
    }

    StringBuilder msgBuilder;
    if (expectHashes.getCount() != resultHashes.getCount())
    {
        msgBuilder << "The number of hashes doesn't match, expect: " << expectHashes.getCount()
                   << ", actual: " << resultHashes.getCount() << "\n";
        getTestReporter()->message(TestMessageType::TestFailure, msgBuilder.toString().getBuffer());
        return SLANG_FAIL;
    }

    for (Index i = 0; i < expectHashes.getCount(); i++)
    {
        if (expectHashes[i].targetIndex != resultHashes[i].targetIndex)
        {
            msgBuilder << "Failed to match 'targetIndex' at index " << i << "\n";
            msgBuilder << "Expect: " << expectHashes[i].targetIndex
                       << ", actual: " << resultHashes[i].targetIndex << "\n";
            getTestReporter()->message(
                TestMessageType::TestFailure,
                msgBuilder.toString().getBuffer());
            return SLANG_FAIL;
        }
        if (expectHashes[i].entryPointIndex != resultHashes[i].entryPointIndex)
        {
            msgBuilder << "Failed to match 'entryPointIndex' at index " << i << "\n";
            msgBuilder << "Expect: " << expectHashes[i].entryPointIndex
                       << ", actual: " << resultHashes[i].entryPointIndex << "\n";
            getTestReporter()->message(
                TestMessageType::TestFailure,
                msgBuilder.toString().getBuffer());
            return SLANG_FAIL;
        }

        if (expectHashes[i].hash != resultHashes[i].hash)
        {
            msgBuilder << "Failed to match 'hash' at index " << i << "\n";
            msgBuilder << "Expect: " << expectHashes[i].hash << ", actual: " << resultHashes[i].hash
                       << "\n";
            getTestReporter()->message(
                TestMessageType::TestFailure,
                msgBuilder.toString().getBuffer());
            return SLANG_FAIL;
        }
    }

    return SLANG_OK;
}

static SlangResult cleanupRecordFiles()
{
    SlangResult res = Path::removeNonEmpty("slang-record");
    if (SLANG_FAILED(res))
    {
        getTestReporter()->message(
            TestMessageType::TestFailure,
            "Failed to remove 'slang-record' directory\n");
    }

    return res;
}

static SlangResult runTest(UnitTestContext* context, const char* testName)
{
    List<entryHashInfo> expectHashes;
    List<entryHashInfo> resultHashes;
    SlangResult res = SLANG_OK;
    if ((res = runExample(context, testName, expectHashes)) != SLANG_OK)
    {
        goto error;
    }

    if ((res = replayExample(context, resultHashes)) != SLANG_OK)
    {
        goto error;
    }

    if ((res = resultCompare(expectHashes, resultHashes)) != SLANG_OK)
    {
        goto error;
    }

error:
    cleanupRecordFiles();
    return res;
}

static SlangResult runTests(UnitTestContext* context)
{
    const char* testBinaryNames[] = {
        "cpu-hello-world",
        "triangle",
        "ray-tracing",
        "ray-tracing-pipeline",
        "autodiff-texture",
        "gpu-printing"
        // "shader-object", // these examples requires reflection API to replay, we have to disable
        // it for now. "model-viewer",
    };

    SlangResult finalRes = SLANG_OK;
    for (const auto& testBinaryName : testBinaryNames)
    {
        SlangResult res = runTest(context, testBinaryName);
        if (SLANG_FAILED(res))
        {
            StringBuilder msgBuilder;
            msgBuilder << "Failed subtest: '" << testBinaryName << "'\n\n\n";
            getTestReporter()->message(
                TestMessageType::TestFailure,
                msgBuilder.toString().getBuffer());
            finalRes = res;
        }
    }

    return finalRes;
}

// Those examples all depend on the Vulkan, so we only run them on non-Apple platforms.
// In the future, we may be able to modify the examples further to remove all the render APIs
// such that it can be ran on Apple platforms.
#if !(SLANG_APPLE_FAMILY)
SLANG_UNIT_TEST(RecordReplay)
{
    SLANG_CHECK(SLANG_SUCCEEDED(runTests(unitTestContext)));
}

#endif
