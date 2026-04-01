// unit-test-command-line-args.cpp

#include "../../source/compiler-core/slang-command-line-args.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

SLANG_UNIT_TEST(commandLineArgs)
{
    RefPtr<CommandLineContext> context = new CommandLineContext;


    // Simple scoped version
    {
        CommandLineArgs args(context);
        DownstreamArgs downstreamArgs(context);

        DiagnosticSink sink(context->getSourceManager(), nullptr);

        const char* inArgs[] = {
            "-Xa...",
            "-blah",
            "10",
            "-X.",
        };

        args.setArgs(inArgs, SLANG_COUNT_OF(inArgs));

        SLANG_CHECK(SLANG_SUCCEEDED(
            downstreamArgs.stripDownstreamArgs(args, DownstreamArgs::Flag::AllowNewNames, &sink)));

        const char* aArgs[] = {"-blah", "10"};

        SLANG_CHECK(downstreamArgs.getArgsByName("a").hasArgs(aArgs, SLANG_COUNT_OF(aArgs)));
        SLANG_CHECK(args.getArgCount() == 0 && sink.getErrorCount() == 0);
    }

    // Leaving off terminating -X. is ok
    {
        CommandLineArgs args(context);
        DownstreamArgs downstreamArgs(context);

        DiagnosticSink sink(context->getSourceManager(), nullptr);

        const char* inArgs[] = {
            "-Xa...",
            "-blah",
            "10",
        };

        args.setArgs(inArgs, SLANG_COUNT_OF(inArgs));

        SLANG_CHECK(SLANG_SUCCEEDED(
            downstreamArgs.stripDownstreamArgs(args, DownstreamArgs::Flag::AllowNewNames, &sink)));

        const char* aArgs[] = {"-blah", "10"};

        SLANG_CHECK(downstreamArgs.getArgsByName("a").hasArgs(aArgs, SLANG_COUNT_OF(aArgs)));
        SLANG_CHECK(args.getArgCount() == 0 && sink.getErrorCount() == 0);
    }

    // Having a nesting

    {
        CommandLineArgs args(context);
        DownstreamArgs downstreamArgs(context);

        DiagnosticSink sink(context->getSourceManager(), nullptr);

        const char* inArgs[] = {
            "-something",
            "andAnother",
            "-Xa...",
            "-blah",
            "-Xb...",
            "-hey",
            "-X.",
            "10",
            "-X.",
            "-Xc",
            "somethingForC",
        };

        args.setArgs(inArgs, SLANG_COUNT_OF(inArgs));

        SLANG_CHECK(SLANG_SUCCEEDED(
            downstreamArgs.stripDownstreamArgs(args, DownstreamArgs::Flag::AllowNewNames, &sink)));

        const char* aArgs[] = {"-blah", "-Xb...", "-hey", "-X.", "10"};

        const char* cArgs[] = {
            "somethingForC",
        };

        const char* mainArgs[] = {
            "-something",
            "andAnother",
        };

        SLANG_CHECK(downstreamArgs.getArgsByName("a").hasArgs(aArgs, SLANG_COUNT_OF(aArgs)));
        SLANG_CHECK(downstreamArgs.getArgsByName("c").hasArgs(cArgs, SLANG_COUNT_OF(cArgs)));

        SLANG_CHECK(args.hasArgs(mainArgs, SLANG_COUNT_OF(mainArgs)) && sink.getErrorCount() == 0);
    }
}
