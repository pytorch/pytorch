// test-context.cpp
#include "options.h"

#include "../../source/core/slang-io.h"
#include "../../source/core/slang-string-util.h"

#include <stdio.h>
#include <stdlib.h>

using namespace Slang;

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!! CategorySet !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

TestCategory* TestCategorySet::add(String const& name, TestCategory* parent)
{
    RefPtr<TestCategory> category(new TestCategory);
    category->name = name;
    category->parent = parent;

    m_categoryMap.add(name, category);
    return category;
}

TestCategory* TestCategorySet::find(String const& name)
{
    if (auto category = m_categoryMap.tryGetValue(name))
    {
        return category->Ptr();
    }
    return nullptr;
}

TestCategory* TestCategorySet::findOrError(String const& name)
{
    TestCategory* category = find(name);
    if (!category)
    {
        StdWriters::getError().print("error: unknown test category name '%s'\n", name.getBuffer());
    }
    return category;
}

/* We need a way to differentiate a subCommand from say a test prefix. Here
we assume a command is just alpha characters or -, and this would differentiate it from
typical prefix usage (which is generally a directory). */
static bool _isSubCommand(const char* arg)
{
    for (; *arg; arg++)
    {
        const char c = *arg;
        // A command is just letters
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '-'))
        {
            return false;
        }
    }
    return true;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!! Options !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* static */ void Options::showHelp(WriterHelper stdOut)
{
    stdOut.print(
        "Usage: slang-test [options] [test-prefix...]\n"
        "\n"
        "Options:\n"
        "  -h, --help                     Show this help message\n"
        "  -bindir <path>                 Set directory for binaries (default: the path to the "
        "slang-test executable)\n"
        "  -test-dir <path>               Set directory for test files (default: tests/)\n"
        "  -v                             Enable verbose output\n"
        "  -hide-ignored                  Hide results from ignored tests\n"
        "  -api-only                      Only run tests that use specified APIs\n"
        "  -verbose-paths                 Use verbose paths in output\n"
        "  -category <name>               Only run tests in specified category\n"
        "  -exclude <name>                Exclude tests in specified category\n"
        "  -api <expr>                    Enable specific APIs (e.g., 'vk+dx12' or '+dx11')\n"
        "  -synthesizedTestApi <expr>     Set APIs for synthesized tests\n"
        "  -skip-api-detection            Skip API availability detection\n"
        "  -server-count <n>              Set number of test servers (default: 1)\n"
        "  -show-adapter-info             Show detailed adapter information\n"
        "  -generate-hlsl-baselines       Generate HLSL test baselines\n"
        "  -skip-reference-image-generation Skip generating reference images for render tests\n"
        "  -emit-spirv-via-glsl           Emit SPIR-V through GLSL instead of directly\n"
        "  -expected-failure-list <file>  Specify file containing expected failures\n"
        "  -use-shared-library            Run tests in-process using shared library\n"
        "  -use-test-server               Run tests using test server\n"
        "  -use-fully-isolated-test-server  Run each test in isolated server\n"
        "\n"
        "Output modes:\n"
        "  -appveyor                      Use AppVeyor output format\n"
        "  -travis                        Use Travis CI output format\n"
        "  -teamcity                      Use TeamCity output format\n"
        "  -xunit                         Use xUnit output format\n"
        "  -xunit2                        Use xUnit 2 output format\n"
        "\n"
        "Test prefixes are used to filter which tests to run. If no prefix is specified,\n"
        "all tests will be run.\n");
}

/* static */ Result Options::parse(
    int argc,
    char** argv,
    TestCategorySet* categorySet,
    Slang::WriterHelper stdError,
    Options* optionsOut)
{
    // Reset the options
    *optionsOut = Options();

    List<const char*> positionalArgs;

    int argCount = argc;
    char const* const* argCursor = argv;
    char const* const* argEnd = argCursor + argCount;

    // first argument is the application name
    if (argCursor != argEnd)
    {
        optionsOut->appName = *argCursor++;
    }

    // Check for help flags first
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            showHelp(stdError);
            return SLANG_FAIL;
        }
    }

    // now iterate over arguments to collect options
    while (argCursor != argEnd)
    {
        char const* arg = *argCursor++;

        if (arg[0] != '-')
        {
            // We need to determine if this is a command, the confusion is that
            // previously we can specify a test prefix as just a single positional arg.
            // To rule this out, here it can only be a subCommand if it is just text

            if (_isSubCommand(arg))
            {
                optionsOut->subCommand = arg;
                // Make the first arg the command name
                optionsOut->subCommandArgs.add(optionsOut->subCommand);

                // Add all the remaining commands to subCommands
                for (; argCursor != argEnd; ++argCursor)
                {
                    optionsOut->subCommandArgs.add(*argCursor);
                }
                // Done
                return SLANG_OK;
            }

            positionalArgs.add(arg);
            continue;
        }

        if (strcmp(arg, "--") == 0)
        {
            // Add all positional args at the end
            while (argCursor != argEnd)
            {
                positionalArgs.add(*argCursor++);
            }
            break;
        }

        if (strcmp(arg, "-bindir") == 0)
        {
            if (argCursor == argEnd)
            {
                stdError.print("error: expected operand for '%s'\n", arg);
                showHelp(stdError);
                return SLANG_FAIL;
            }
            optionsOut->binDir = *argCursor++;
        }
        else if (strcmp(arg, "-use-shared-library") == 0)
        {
            optionsOut->defaultSpawnType = SpawnType::UseSharedLibrary;
        }
        else if (strcmp(arg, "-use-test-server") == 0)
        {
            optionsOut->defaultSpawnType = SpawnType::UseTestServer;
        }
        else if (strcmp(arg, "-use-fully-isolated-test-server") == 0)
        {
            optionsOut->defaultSpawnType = SpawnType::UseFullyIsolatedTestServer;
        }
        else if (strcmp(arg, "-v") == 0)
        {
            optionsOut->shouldBeVerbose = true;
        }
        else if (strcmp(arg, "-hide-ignored") == 0)
        {
            optionsOut->hideIgnored = true;
        }
        else if (strcmp(arg, "-api-only") == 0)
        {
            optionsOut->apiOnly = true;
        }
        else if (strcmp(arg, "-verbose-paths") == 0)
        {
            optionsOut->verbosePaths = true;
        }
        else if (strcmp(arg, "-generate-hlsl-baselines") == 0)
        {
            optionsOut->generateHLSLBaselines = true;
        }
        else if (strcmp(arg, "-release") == 0)
        {
            // Assumed to be handle by .bat file that called us
        }
        else if (strcmp(arg, "-debug") == 0)
        {
            // Assumed to be handle by .bat file that called us
        }
        else if (strcmp(arg, "-configuration") == 0)
        {
            if (argCursor == argEnd)
            {
                stdError.print("error: expected operand for '%s'\n", arg);
                showHelp(stdError);
                return SLANG_FAIL;
            }
            argCursor++;
            // Assumed to be handle by .bat file that called us
        }
        else if (strcmp(arg, "-platform") == 0)
        {
            if (argCursor == argEnd)
            {
                stdError.print("error: expected operand for '%s'\n", arg);
                showHelp(stdError);
                return SLANG_FAIL;
            }
            argCursor++;
            // Assumed to be handle by .bat file that called us
        }
        else if (strcmp(arg, "-server-count") == 0)
        {
            if (argCursor == argEnd)
            {
                stdError.print("error: expected operand for '%s'\n", arg);
                showHelp(stdError);
                return SLANG_FAIL;
            }
            optionsOut->serverCount = stringToInt(*argCursor++);
            if (optionsOut->serverCount <= 0)
            {
                optionsOut->serverCount = 1;
            }
        }
        else if (strcmp(arg, "-appveyor") == 0)
        {
            optionsOut->outputMode = TestOutputMode::AppVeyor;
            optionsOut->dumpOutputOnFailure = true;
        }
        else if (strcmp(arg, "-travis") == 0)
        {
            optionsOut->outputMode = TestOutputMode::Travis;
            optionsOut->dumpOutputOnFailure = true;
        }
        else if (strcmp(arg, "-xunit") == 0)
        {
            optionsOut->outputMode = TestOutputMode::XUnit;
        }
        else if (strcmp(arg, "-xunit2") == 0)
        {
            optionsOut->outputMode = TestOutputMode::XUnit2;
        }
        else if (strcmp(arg, "-teamcity") == 0)
        {
            optionsOut->outputMode = TestOutputMode::TeamCity;
        }
        else if (strcmp(arg, "-category") == 0)
        {
            if (argCursor == argEnd)
            {
                stdError.print("error: expected operand for '%s'\n", arg);
                showHelp(stdError);
                return SLANG_FAIL;
            }
            auto category = categorySet->findOrError(*argCursor++);
            if (category)
            {
                optionsOut->includeCategories.add(category, category);
            }
        }
        else if (strcmp(arg, "-exclude") == 0)
        {
            if (argCursor == argEnd)
            {
                stdError.print("error: expected operand for '%s'\n", arg);
                showHelp(stdError);
                return SLANG_FAIL;
            }
            auto category = categorySet->findOrError(*argCursor++);
            if (category)
            {
                optionsOut->excludeCategories.add(category, category);
            }
        }
        else if (strcmp(arg, "-api") == 0)
        {
            if (argCursor == argEnd)
            {
                stdError.print(
                    "error: expecting an api expression (eg 'vk+dx12' or '+dx11') '%s'\n",
                    arg);
                showHelp(stdError);
                return SLANG_FAIL;
            }
            const char* apiList = *argCursor++;

            SlangResult res = RenderApiUtil::parseApiFlags(
                UnownedStringSlice(apiList),
                optionsOut->enabledApis,
                &optionsOut->enabledApis);
            if (SLANG_FAILED(res))
            {
                stdError.print("error: unable to parse api expression '%s'\n", apiList);
                return res;
            }
        }
        else if (strcmp(arg, "-synthesizedTestApi") == 0)
        {
            if (argCursor == argEnd)
            {
                stdError.print(
                    "error: expected an api expression (eg 'vk+dx12' or '+dx11') '%s'\n",
                    arg);
                showHelp(stdError);
                return SLANG_FAIL;
            }
            const char* apiList = *argCursor++;

            SlangResult res = RenderApiUtil::parseApiFlags(
                UnownedStringSlice(apiList),
                optionsOut->synthesizedTestApis,
                &optionsOut->synthesizedTestApis);
            if (SLANG_FAILED(res))
            {
                stdError.print("error: unable to parse api expression '%s'\n", apiList);
                return res;
            }
        }
        else if (strcmp(arg, "-skip-api-detection") == 0)
        {
            optionsOut->skipApiDetection = true;
        }
        else if (strcmp(arg, "-emit-spirv-via-glsl") == 0)
        {
            optionsOut->emitSPIRVDirectly = false;
        }
        else if (strcmp(arg, "-expected-failure-list") == 0)
        {
            if (argCursor == argEnd)
            {
                stdError.print("error: expected operand for '%s'\n", arg);
                showHelp(stdError);
                return SLANG_FAIL;
            }
            auto fileName = *argCursor++;
            String text;
            File::readAllText(fileName, text);
            List<UnownedStringSlice> lines;
            StringUtil::split(text.getUnownedSlice(), '\n', lines);
            for (auto line : lines)
            {
                optionsOut->expectedFailureList.add(line);
            }
        }
        else if (strcmp(arg, "-test-dir") == 0)
        {
            if (argCursor == argEnd)
            {
                stdError.print("error: expected operand for '%s'\n", arg);
                showHelp(stdError);
                return SLANG_FAIL;
            }
            optionsOut->testDir = *argCursor++;
        }
        else if (strcmp(arg, "-show-adapter-info") == 0)
        {
            optionsOut->showAdapterInfo = true;
        }
        else if (strcmp(arg, "-skip-reference-image-generation") == 0)
        {
            optionsOut->skipReferenceImageGeneration = true;
        }
        else
        {
            stdError.print("unknown option '%s'\n", arg);
            showHelp(stdError);
            return SLANG_FAIL;
        }
    }

    {
        // Find out what apis are available
        const int availableApis = RenderApiUtil::getAvailableApis();
        // Only allow apis we know are available
        optionsOut->enabledApis &= availableApis;

        // Can only synth for apis that are available
        optionsOut->synthesizedTestApis &= optionsOut->enabledApis;
    }


    // first positional argument is source shader path
    optionsOut->testPrefixes.clear();
    optionsOut->testPrefixes.reserve(positionalArgs.getCount());
    for (auto testPrefix : positionalArgs)
    {
        Slang::StringBuilder sb;
        Slang::Path::simplify(testPrefix, Slang::Path::SimplifyStyle::NoRoot, sb);
        optionsOut->testPrefixes.add(sb);
    }

    if (optionsOut->binDir.getLength() == 0)
    {
        // If the binDir isn't set try using the path to the executable
        String exePath = Path::getExecutablePath();
        if (exePath.getLength())
        {
            optionsOut->binDir = Path::getParentDirectory(exePath);
        }
    }

    if (optionsOut->testDir.getLength() == 0)
    {
        // If the test directory isn't set, use the "tests" directory
        optionsOut->testDir = String("tests/");
    }

    return SLANG_OK;
}
