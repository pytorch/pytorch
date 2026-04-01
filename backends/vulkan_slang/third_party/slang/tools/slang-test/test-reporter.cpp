// test-reporter.cpp
#include "test-reporter.h"

#include "../../source/core/slang-process-util.h"
#include "../../source/core/slang-string-util.h"

#include <mutex>
#include <stdio.h>
#include <stdlib.h>

using namespace Slang;

/* static */ TestReporter* TestReporter::s_reporter = nullptr;

static void appendXmlEncode(char c, StringBuilder& out)
{
    switch (c)
    {
    case '&':
        out << "&amp;";
        break;
    case '<':
        out << "&lt;";
        break;
    case '>':
        out << "&gt;";
        break;
    case '\'':
        out << "&apos;";
        break;
    case '"':
        out << "&quot;";
        break;
    default:
        out.append(c);
    }
}

static bool isXmlEncodeChar(char c)
{
    switch (c)
    {
    case '&':
    case '<':
    case '>':
        {
            return true;
        }
    }
    return false;
}

static void appendXmlEncode(const String& in, StringBuilder& out)
{
    const char* cur = in.getBuffer();
    const char* end = cur + in.getLength();

    while (cur < end)
    {
        const char* start = cur;
        // Look for a run of non encoded
        while (cur < end && !isXmlEncodeChar(*cur))
        {
            cur++;
        }
        // Write it
        if (cur > start)
        {
            out.append(start, UInt(end - start));
        }

        // if not at the end, we must be on an xml encoded character, so just output it xml encoded.
        if (cur < end)
        {
            const char encodeChar = *cur++;
            assert(isXmlEncodeChar(encodeChar));
            appendXmlEncode(encodeChar, out);
        }
    }
}

TestReporter::TestReporter()
    : m_outputMode(TestOutputMode::Default)
{
    m_totalTestCount = 0;
    m_passedTestCount = 0;
    m_failedTestCount = 0;
    m_ignoredTestCount = 0;
    m_expectedFailedTestCount = 0;
    m_maxFailTestResults = 10;

    m_inTest = false;
    m_dumpOutputOnFailure = false;
    m_isVerbose = false;
}

Result TestReporter::init(
    TestOutputMode outputMode,
    const HashSet<String>& expectedFailureList,
    bool isSubReporter)
{
    m_outputMode = outputMode;
    m_isSubReporter = isSubReporter;
    m_expectedFailureList = expectedFailureList;
    return SLANG_OK;
}

TestReporter::~TestReporter() {}

bool TestReporter::canWriteStdError() const
{
    switch (m_outputMode)
    {
    case TestOutputMode::XUnit:
    case TestOutputMode::XUnit2:
        {
            return false;
        }
    default:
        return true;
    }
}

void TestReporter::startTest(const char* testName)
{
    // Must be in a suite
    assert(m_suiteStack.getCount());
    assert(!m_inTest);

    m_inTest = true;

    m_numCurrentResults = 0;
    m_numFailResults = 0;

    m_currentInfo = TestInfo();
    m_currentInfo.name = testName;
    m_currentMessage.clear();
}

void TestReporter::endTest()
{
    assert(m_suiteStack.getCount());
    assert(m_inTest);

    m_currentInfo.message = m_currentMessage;

    _addResult(m_currentInfo);

    m_inTest = false;
}

void TestReporter::addResult(TestResult result)
{
    assert(m_inTest);

    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (result == TestResult::Fail && m_expectedFailureList.contains(m_currentInfo.name))
        result = TestResult::ExpectedFail;
    m_currentInfo.testResult = combine(m_currentInfo.testResult, result);
    m_numCurrentResults++;
}

TestResult TestReporter::getResult() const
{
    return m_currentInfo.testResult;
}

void TestReporter::addExecutionTime(double time)
{
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    m_currentInfo.executionTime = time;
}

void TestReporter::addResultWithLocation(
    TestResult result,
    const char* testText,
    const char* file,
    int line)
{
    assert(m_inTest);

    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    result = adjustResult(m_currentInfo.name.getUnownedSlice(), result);

    m_numCurrentResults++;

    m_currentInfo.testResult = combine(m_currentInfo.testResult, result);
    if (result != TestResult::Fail)
    {
        // We don't need to output the result if it
        return;
    }

    m_numFailResults++;

    if (m_maxFailTestResults > 0)
    {
        if (m_numFailResults > m_maxFailTestResults)
        {
            if (m_numFailResults == m_maxFailTestResults + 1)
            {
                // It's a failure, but to show that there are more than are going to be shown, just
                // show '...'
                message(TestMessageType::TestFailure, "...");
            }
            return;
        }
    }

    StringBuilder buf;
    buf << testText << " - " << file << " (" << line << ")";

    message(TestMessageType::TestFailure, buf);
}

void TestReporter::addResultWithLocation(
    bool testSucceeded,
    const char* testText,
    const char* file,
    int line)
{
    addResultWithLocation(
        testSucceeded ? TestResult::Pass : TestResult::Fail,
        testText,
        file,
        line);
}

TestResult TestReporter::addTest(const String& testName, bool isPass)
{
    const TestResult res = isPass ? TestResult::Pass : TestResult::Fail;
    addTest(testName, res);
    return res;
}

void TestReporter::consolidateWith(TestReporter* other)
{
    m_testInfos.addRange(other->m_testInfos);
    m_failedTestCount += other->m_failedTestCount;
    m_ignoredTestCount += other->m_ignoredTestCount;
    m_passedTestCount += other->m_passedTestCount;
    m_expectedFailedTestCount += other->m_expectedFailedTestCount;
    m_totalTestCount += other->m_totalTestCount;
}

void TestReporter::dumpOutputDifference(const String& expectedOutput, const String& actualOutput)
{
    StringBuilder builder;

    StringUtil::appendFormat(
        builder,
        "ERROR:\n"
        "EXPECTED{{{\n%s}}}\n"
        "ACTUAL{{{\n%s}}}\n",
        expectedOutput.getBuffer(),
        actualOutput.getBuffer());

    // Add to the m_currentInfo
    message(TestMessageType::TestFailure, builder);
}

static char _getTeamCityEscapeChar(char c)
{
    switch (c)
    {
    case '|':
        return '|';
    case '\'':
        return '\'';
    case '\n':
        return 'n';
    case '\r':
        return 'r';
    case '[':
        return '[';
    case ']':
        return ']';
    default:
        return 0;
    }
}

static void _appendEncodedTeamCityString(const UnownedStringSlice& in, StringBuilder& builder)
{
    const char* start = in.begin();
    const char* cur = start;
    const char* end = in.end();

    for (const char* cur = start; cur < end; cur++)
    {
        const char c = *cur;
        const char escapeChar = _getTeamCityEscapeChar(c);
        if (escapeChar)
        {
            // Flush
            if (cur > start)
            {
                builder.append(start, UInt(cur - start));
            }

            builder.append('|');
            builder.append(escapeChar);
            start = cur + 1;
        }
    }

    // Flush the end
    if (end > start)
    {
        builder.append(start, UInt(end - start));
    }
}

static void _appendTime(double timeInSec, StringBuilder& out)
{
    SLANG_ASSERT(timeInSec >= 0.0);
    if (timeInSec == 0.0 || timeInSec >= 1.0)
    {
        out << timeInSec << "s";
        return;
    }
    timeInSec *= 1000.0f;
    if (timeInSec > 1.0f)
    {
        out << timeInSec << "ms";
        return;
    }
    timeInSec *= 1000.0f;
    if (timeInSec > 1.0f)
    {
        out << timeInSec << "us";
        return;
    }

    timeInSec *= 1000.0f;
    out << timeInSec << "ns";
}

void TestReporter::_addResult(TestInfo info)
{
    if (info.testResult == TestResult::Ignored && m_hideIgnored)
    {
        return;
    }
    info.testResult = adjustResult(info.name.getUnownedSlice(), info.testResult);

    m_totalTestCount++;

    switch (info.testResult)
    {
    case TestResult::Fail:
        m_failedTestCount++;
        break;

    case TestResult::Pass:
        m_passedTestCount++;
        break;
    case TestResult::ExpectedFail:
        m_expectedFailedTestCount++;
        break;

    case TestResult::Ignored:
        m_ignoredTestCount++;
        break;

    default:
        assert(!"unexpected");
        break;
    }

    m_testInfos.add(info);

    auto defaultOutputFunc = [](const TestInfo& info)
    {
        char const* resultString = "UNEXPECTED";
        switch (info.testResult)
        {
        case TestResult::Fail:
            resultString = "FAILED";
            break;
        case TestResult::ExpectedFail:
            resultString = "failed(expected)";
            break;
        case TestResult::Pass:
            resultString = "passed";
            break;
        case TestResult::Ignored:
            resultString = "ignored";
            break;
        default:
            assert(!"unexpected");
            break;
        }

        StringBuilder buffer;
        if (info.executionTime > 0.0f)
        {
            _appendTime(info.executionTime, buffer);
        }
        printf(
            "%s test: '%S' %s\n",
            resultString,
            info.name.toWString().begin(),
            buffer.getBuffer());
        fflush(stdout);
    };

    switch (m_outputMode)
    {
    default:
        {
            defaultOutputFunc(info);
            break;
        }
    case TestOutputMode::TeamCity:
        {
            StringBuilder escapedTestName;
            _appendEncodedTeamCityString(info.name.getUnownedSlice(), escapedTestName);

            printf("##teamcity[testStarted name='%s']\n", escapedTestName.begin());

            switch (info.testResult)
            {
            case TestResult::Fail:
                {
                    if (info.message.getLength())
                    {
                        StringBuilder escapedMessage;
                        _appendEncodedTeamCityString(
                            info.message.getUnownedSlice(),
                            escapedMessage);
                        printf(
                            "##teamcity[testFailed name='%s' message='%s']\n",
                            escapedTestName.begin(),
                            escapedMessage.begin());
                    }
                    else
                    {
                        printf("##teamcity[testFailed name='%s']\n", escapedTestName.begin());
                    }
                    break;
                }
            case TestResult::Pass:
            case TestResult::ExpectedFail:
                {
                    StringBuilder message;
                    message << info.message;
                    // Add execution time if one is set
                    if (info.executionTime > 0.0)
                    {
                        if (message.getLength())
                        {
                            message << " ";
                        }
                        _appendTime(info.executionTime, message);
                    }

                    if (message.getLength())
                    {
                        StringBuilder escapedMessage;
                        _appendEncodedTeamCityString(message.getUnownedSlice(), escapedMessage);
                        printf(
                            "##teamcity[testStdOut name='%s' out='%s']\n",
                            escapedTestName.begin(),
                            escapedMessage.begin());
                    }
                    break;
                }
            case TestResult::Ignored:
                {
                    if (info.message.getLength())
                    {
                        StringBuilder escapedMessage;
                        _appendEncodedTeamCityString(
                            info.message.getUnownedSlice(),
                            escapedMessage);

                        printf(
                            "##teamcity[testIgnored name='%s' message='%s']\n",
                            escapedTestName.begin(),
                            escapedMessage.begin());
                    }
                    else
                    {
                        printf("##teamcity[testIgnored name='%s']\n", escapedTestName.begin());
                    }
                    break;
                }
            default:
                assert(!"unexpected");
                break;
            }

            printf("##teamcity[testFinished name='%s']\n", escapedTestName.begin());
            fflush(stdout);
            break;
        }
    case TestOutputMode::XUnit2:
    case TestOutputMode::XUnit:
        {
            // Don't output anything -> we'll output all in one go at the end
            break;
        }
    case TestOutputMode::AppVeyor:
        {
            char const* resultString = "None";
            switch (info.testResult)
            {
            case TestResult::Fail:
                resultString = "Failed";
                break;
            case TestResult::Pass:
                resultString = "Passed";
                break;
            case TestResult::Ignored:
                resultString = "Ignored";
                break;
            case TestResult::ExpectedFail:
                resultString = "ExpectedFail";
                break;

            default:
                assert(!"unexpected");
                break;
            }

            // https://www.appveyor.com/docs/build-worker-api/#add-tests

            CommandLine cmdLine;
            cmdLine.setExecutableLocation(ExecutableLocation("appveyor"));
            cmdLine.addArg("AddTest");
            cmdLine.addArg(info.name);
            cmdLine.addArg("-FileName");
            // TODO: this isn't actually a file name in all cases
            cmdLine.addArg(info.name);
            cmdLine.addArg("-Framework");
            cmdLine.addArg("slang-test");
            cmdLine.addArg("-Outcome");
            cmdLine.addArg(resultString);

            // If has execution time output it
            if (info.executionTime > 0.0)
            {
                StringBuilder builder;
                _appendTime(info.executionTime, builder);
                cmdLine.addArg("-StdOut");
                cmdLine.addArg(builder);
            }

            ExecuteResult exeRes;
            SlangResult res = ProcessUtil::execute(cmdLine, exeRes);

            if (SLANG_FAILED(res))
            {
                messageFormat(
                    TestMessageType::Info,
                    "failed to add appveyor test results for '%S'\n",
                    info.name.toWString().begin());

#if 0
                String cmdLineString = ProcessUtil::getCommandLineString(cmdLine);
                fprintf(stderr, "[%d] TEST RESULT: %s {%d} {%s} {%s}\n", err, cmdLineString.getBuffer(),
                    exeRes.resultCode,
                    exeRes.standardOutput.begin(),
                    exeRes.standardError.begin());
#endif
            }
            defaultOutputFunc(info);
            break;
        }
    }
}

void TestReporter::addTest(const String& testName, TestResult testResult)
{
    // Can't add this way if in test
    assert(!m_inTest);

    TestInfo info;
    info.name = testName;
    info.testResult = testResult;
    _addResult(info);
}

void TestReporter::message(TestMessageType type, const String& message)
{
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    if (type == TestMessageType::Info)
    {
        if (m_isVerbose && canWriteStdError())
        {
            fputs(message.getBuffer(), stderr);
        }
        fflush(stderr);
        // Just dump out if can dump out
        return;
    }

    if (canWriteStdError())
    {
        if (type == TestMessageType::RunError || type == TestMessageType::TestFailure)
        {
            fprintf(stderr, "error: ");
            fputs(message.getBuffer(), stderr);
            fprintf(stderr, "\n");
        }
        else
        {
            fputs(message.getBuffer(), stderr);
        }
        fflush(stderr);
    }

    if (m_currentMessage.getLength() > 0)
    {
        m_currentMessage << "\n";
    }
    m_currentMessage.append(message);
}

void TestReporter::messageFormat(TestMessageType type, char const* format, ...)
{
    StringBuilder builder;

    va_list args;
    va_start(args, format);
    StringUtil::append(format, args, builder);
    va_end(args);

    message(type, builder);
}

void TestReporter::message(TestMessageType type, const char* messageContent)
{
    message(type, String(messageContent));
}


bool TestReporter::didAllSucceed() const
{
    return m_failedTestCount == 0;
}

void TestReporter::outputSummary()
{
    auto passCount = m_passedTestCount;
    auto rawTotal = m_totalTestCount;
    auto ignoredCount = m_ignoredTestCount;

    auto runTotal = rawTotal - ignoredCount;

    switch (m_outputMode)
    {
    default:
        {
            if (!m_totalTestCount)
            {
                printf("no tests run\n");
                return;
            }

            int percentPassed = 0;
            if (runTotal > 0)
            {
                percentPassed = (passCount * 100) / runTotal;
            }

            printf("\n===\n%d%% of tests passed (%d/%d)", percentPassed, passCount, runTotal);
            if (ignoredCount)
            {
                printf(", %d tests ignored", ignoredCount);
            }
            if (m_expectedFailedTestCount)
            {
                printf(", %d tests failed expectedly", m_expectedFailedTestCount);
                printf("\n===\n\n");
                printf("\npassing tests that are expected to fail:\n");
                printf("---\n");
                for (const auto& testInfo : m_testInfos)
                {
                    if (testInfo.testResult == TestResult::Pass)
                    {
                        if (m_expectedFailureList.contains(testInfo.name))
                        {
                            printf("%s\n", testInfo.name.getBuffer());
                        }
                    }
                }
                printf("---\n");
            }
            printf("\n===\n\n");
            if (m_failedTestCount)
            {
                printf("failing tests:\n");
                printf("---\n");
                for (const auto& testInfo : m_testInfos)
                {
                    if (testInfo.testResult == TestResult::Fail)
                    {
                        printf("%s\n", testInfo.name.getBuffer());
                    }
                }
                printf("---\n");
            }

            break;
        }

    case TestOutputMode::XUnit:
        {
            // xUnit 1.0 format

            printf("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
            printf(
                "<testsuites tests=\"%d\" failures=\"%d\" disabled=\"%d\" errors=\"0\" "
                "name=\"AllTests\">\n",
                m_totalTestCount,
                m_failedTestCount,
                m_ignoredTestCount);
            printf(
                "  <testsuite name=\"all\" tests=\"%d\" failures=\"%d\" disabled=\"%d\" "
                "errors=\"0\" time=\"0\">\n",
                m_totalTestCount,
                m_failedTestCount,
                m_ignoredTestCount);

            for (const auto& testInfo : m_testInfos)
            {
                const int numFailed = (testInfo.testResult == TestResult::Fail);
                const int numIgnored = (testInfo.testResult == TestResult::Ignored);
                // int numPassed = (testInfo.testResult == TestResult::ePass);

                if (testInfo.testResult == TestResult::Pass)
                {
                    printf(
                        "    <testcase name=\"%s\" status=\"run\"/>\n",
                        testInfo.name.getBuffer());
                }
                else
                {
                    printf(
                        "    <testcase name=\"%s\" status=\"run\">\n",
                        testInfo.name.getBuffer());
                    switch (testInfo.testResult)
                    {
                    case TestResult::Fail:
                        {
                            StringBuilder buf;
                            appendXmlEncode(testInfo.message, buf);

                            printf("      <error>\n");
                            printf("%s", buf.getBuffer());
                            printf("      </error>\n");
                            break;
                        }
                    case TestResult::Ignored:
                        {
                            printf("      <skip>Ignored</skip>\n");
                            break;
                        }
                    default:
                        break;
                    }
                    printf("    </testcase>\n");
                }
            }

            printf("  </testsuite>\n");
            printf("</testSuites>\n");
            break;
        }
    case TestOutputMode::XUnit2:
        {
            // https://xunit.github.io/docs/format-xml-v2
            assert("Not currently supported");
            break;
        }
    case TestOutputMode::TeamCity:
        {
            // Don't output a summary
            break;
        }
    }
}

void TestReporter::startSuite(const String& name)
{
    m_suiteStack.add(name);

    switch (m_outputMode)
    {
    case TestOutputMode::TeamCity:
        {
            if (!m_isSubReporter)
            {
                StringBuilder escapedSuiteName;
                _appendEncodedTeamCityString(name.getUnownedSlice(), escapedSuiteName);
                printf("##teamcity[testSuiteStarted name='%s']\n", escapedSuiteName.begin());
            }
            break;
        }
    default:
        break;
    }
}

void TestReporter::endSuite()
{
    assert(m_suiteStack.getCount());

    switch (m_outputMode)
    {
    case TestOutputMode::TeamCity:
        {
            if (!m_isSubReporter)
            {
                const String& name = m_suiteStack.getLast();
                StringBuilder escapedSuiteName;
                _appendEncodedTeamCityString(name.getUnownedSlice(), escapedSuiteName);
                printf("##teamcity[testSuiteFinished name='%s']\n", escapedSuiteName.begin());
            }
            break;
        }
    default:
        break;
    }

    m_suiteStack.removeLast();
}

TestResult TestReporter::adjustResult(UnownedStringSlice testName, TestResult result)
{
    if (result == TestResult::Fail && m_expectedFailureList.contains(testName))
        result = TestResult::ExpectedFail;
    return result;
}
