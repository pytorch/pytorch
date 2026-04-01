// options.h

#ifndef OPTIONS_H_INCLUDED
#define OPTIONS_H_INCLUDED

#include "../../source/core/slang-dictionary.h"
#include "../../source/core/slang-render-api-util.h"
#include "../../source/core/slang-smart-pointer.h"
#include "test-reporter.h"

// A category that a test can be tagged with
struct TestCategory : public Slang::RefObject
{
    // The name of the category, from the user perspective
    Slang::String name;

    // The logical "super-category" of this category
    TestCategory* parent;
};

struct TestCategorySet
{
public:
    /// Find a category with the specified name. Returns nullptr if not found
    TestCategory* find(Slang::String const& name);
    /// Adds a category with the specified name, and parent. Returns the category object.
    /// Parent can be nullptr
    TestCategory* add(Slang::String const& name, TestCategory* parent);
    /// Finds a category by name, else reports and writes an error
    TestCategory* findOrError(Slang::String const& name);

    Slang::RefPtr<TestCategory> defaultCategory; ///< The default category

protected:
    Slang::Dictionary<Slang::String, Slang::RefPtr<TestCategory>> m_categoryMap;
};

enum class SpawnType
{
    Default,          ///< Default - typically uses shared library, on CI may use TestServer
    UseExe,           ///< Tests using executable (for example slangc)
    UseSharedLibrary, ///< Runs testing in process (a crash tan take down the
    UseTestServer,    ///< Use the test server to isolate testing
    UseFullyIsolatedTestServer, ///< Uses a test server for each test (slow!)
};

struct Options
{
    char const* appName = "slang-test";

    // Directory to use when looking for binaries to run. If empty it's not set.
    Slang::String binDir;

    // Root directory to use when looking for test cases
    Slang::String testDir;

    // only run test cases with names have one of these prefixes.
    Slang::List<Slang::String> testPrefixes;

    // generate extra output (notably: command lines we run)
    bool shouldBeVerbose = false;

    // When true results from ignored tests are not shown
    bool hideIgnored = false;

    // When true only tests that use an api that matches the enabledApis flags will run
    bool apiOnly = false;

    // Use verbose paths
    bool verbosePaths = false;

    // force generation of baselines for HLSL tests
    bool generateHLSLBaselines = false;

    // Skip generation of reference images for render tests, assume they already exist
    bool skipReferenceImageGeneration = false;

    // Whether to skip the step of creating test devices to check if an API is actually available.
    bool skipApiDetection = false;

    // Dump expected/actual output on failures, for debugging.
    // This is especially intended for use in continuous
    // integration builds.
    bool dumpOutputOnFailure = false;

    // Set the default spawn type to use
    // Having tests isolated, slows down testing considerably, so using UseSharedLibrary is the most
    // desirable default usually.
    SpawnType defaultSpawnType = SpawnType::Default;

    // kind of output to generate
    TestOutputMode outputMode = TestOutputMode::Default;

    // Only run tests that match one of the given categories
    Slang::Dictionary<TestCategory*, TestCategory*> includeCategories;

    // Exclude test that match one these categories
    Slang::Dictionary<TestCategory*, TestCategory*> excludeCategories;

    // By default we can test against all apis. If is set to anything other than AllOf only tests
    // that *require* the API will be run.
    Slang::RenderApiFlags enabledApis = Slang::RenderApiFlag::AllOf;

    // The subCommand to execute. Will be empty if there is no subCommand
    Slang::String subCommand;

    // Arguments to the sub command. Note that if there is a subCommand the first parameter is
    // always the subCommand itself.
    Slang::List<Slang::String> subCommandArgs;

    // By default we potentially synthesize test for all
    // TODO: Vulkan is disabled by default for now as the majority as vulkan synthesized tests
    // CPU is disabled by default
    // CUDA is disabled by default
    Slang::RenderApiFlags synthesizedTestApis =
        Slang::RenderApiFlag::AllOf & ~(Slang::RenderApiFlag::Vulkan | Slang::RenderApiFlag::CPU);

    // If true, print detailed adapter information
    bool showAdapterInfo = false;

    // Maximum number of test servers to run.
    int serverCount = 1;

    bool emitSPIRVDirectly = true;

    Slang::HashSet<Slang::String> expectedFailureList;

    /// Parse the args, report any errors into stdError, and write the results into optionsOut
    static SlangResult parse(
        int argc,
        char** argv,
        TestCategorySet* categorySet,
        Slang::WriterHelper stdError,
        Options* optionsOut);

    /// Display help message
    static void showHelp(Slang::WriterHelper stdOut);
};

#endif // OPTIONS_H_INCLUDED
