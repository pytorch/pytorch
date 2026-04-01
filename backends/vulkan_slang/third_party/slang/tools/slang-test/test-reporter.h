// test-reporter.h

#ifndef TEST_REPORTER_H_INCLUDED
#define TEST_REPORTER_H_INCLUDED

#include "../../source/core/slang-dictionary.h"
#include "../../source/core/slang-platform.h"
#include "../../source/core/slang-std-writers.h"
#include "../../source/core/slang-string-util.h"
#include "unit-test/slang-unit-test.h"

#include <mutex>

enum class TestOutputMode
{
    Default = 0, ///< Default mode is to write test results to the console
    AppVeyor,    ///< For AppVeyor continuous integration
    Travis,      ///< We currently don't specialize for Travis, but maybe we should.
    XUnit,    ///< xUnit original format  https://nose.readthedocs.io/en/latest/plugins/xunit.html
    XUnit2,   ///< https://xunit.github.io/docs/format-xml-v2
    TeamCity, ///< Output suitable for teamcity
};

class TestReporter : public ITestReporter
{
public:
    struct TestInfo
    {
        TestResult testResult = TestResult::Ignored;
        Slang::String name;
        Slang::String message;      ///< Message that is specific for the testResult
        double executionTime = 0.0; ///< <= 0.0 if not defined. Time is in seconds.
    };

    class TestScope
    {
    public:
        TestScope(TestReporter* reporter, const Slang::String& testName)
            : m_reporter(reporter)
        {
            reporter->startTest(testName.getBuffer());
        }
        ~TestScope() { m_reporter->endTest(); }

    protected:
        TestReporter* m_reporter;
    };

    class SuiteScope
    {
    public:
        SuiteScope(TestReporter* reporter, const Slang::String& suiteName)
            : m_reporter(reporter)
        {
            reporter->startSuite(suiteName);
        }
        ~SuiteScope() { m_reporter->endSuite(); }

    protected:
        TestReporter* m_reporter;
    };

    void startSuite(const Slang::String& name);
    void endSuite();

    TestResult adjustResult(Slang::UnownedStringSlice testName, TestResult result);

    virtual SLANG_NO_THROW void SLANG_MCALL startTest(const char* testName) override;
    virtual SLANG_NO_THROW void SLANG_MCALL addResult(TestResult result) override;
    virtual SLANG_NO_THROW void SLANG_MCALL addResultWithLocation(
        TestResult result,
        const char* testText,
        const char* file,
        int line) override;
    virtual SLANG_NO_THROW void SLANG_MCALL addResultWithLocation(
        bool testSucceeded,
        const char* testText,
        const char* file,
        int line) override;
    virtual SLANG_NO_THROW void SLANG_MCALL addExecutionTime(double time) override;
    virtual SLANG_NO_THROW void SLANG_MCALL endTest() override;

    /// Runs start/endTest and outputs the result
    TestResult addTest(const Slang::String& testName, bool isPass);
    /// Effectively runs start/endTest (so cannot be called inside start/endTest).
    void addTest(const Slang::String& testName, TestResult testResult);

    // Called for an error in the test-runner (not for an error involving a test itself).
    void message(TestMessageType type, const Slang::String& errorText);
    SLANG_ATTR_PRINTF(3, 4)
    void messageFormat(TestMessageType type, char const* message, ...);
    virtual SLANG_NO_THROW void SLANG_MCALL
    message(TestMessageType type, char const* message) override;

    void dumpOutputDifference(
        const Slang::String& expectedOutput,
        const Slang::String& actualOutput);

    void consolidateWith(TestReporter* other);

    /// True if can write output directly to stderr
    bool canWriteStdError() const;


    /// Returns true if all run tests succeeded
    bool didAllSucceed() const;

    /// Returns a result from the current test
    TestResult getResult() const;

    void outputSummary();

    SlangResult init(
        TestOutputMode outputMode,
        const Slang::HashSet<Slang::String>& expectedFailureList,
        bool isSubReporter = false);

    /// Ctor
    TestReporter();
    /// Dtor
    ~TestReporter();

    static TestResult combine(TestResult a, TestResult b) { return (a > b) ? a : b; }

    static TestReporter* get() { return s_reporter; }
    static void set(TestReporter* reporter) { s_reporter = reporter; }

    Slang::List<TestInfo> m_testInfos;

    Slang::List<Slang::String> m_suiteStack;

    int m_totalTestCount;
    int m_passedTestCount;
    int m_failedTestCount;
    int m_ignoredTestCount;
    int m_expectedFailedTestCount;

    int m_maxFailTestResults; ///< Maximum amount of results per test. If 0 it's infinite.

    TestOutputMode m_outputMode = TestOutputMode::Default;
    bool m_dumpOutputOnFailure;
    bool m_isVerbose = false;
    bool m_hideIgnored = false;
    bool m_isSubReporter = false;
    Slang::HashSet<Slang::String> m_expectedFailureList;

protected:
    void _addResult(TestInfo info);

    Slang::StringBuilder m_currentMessage;
    TestInfo m_currentInfo;
    int m_numCurrentResults;
    int m_numFailResults;

    bool m_inTest;

    std::recursive_mutex m_mutex;

    static TestReporter* s_reporter;
};

#endif // TEST_REPORTER_H_INCLUDED
