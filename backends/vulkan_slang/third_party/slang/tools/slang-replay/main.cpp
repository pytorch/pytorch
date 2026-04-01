#include "../../source/core/slang-io.h"

#include <memory>
#include <replay/json-consumer.h>
#include <replay/recordFile-processor.h>
#include <replay/replay-consumer.h>
#include <replay/slang-decoder.h>
#include <stdio.h>

struct Options
{
    bool convertToJson{false};
    Slang::String recordFileName;
};

void printUsage()
{
    printf("Usage: slang-replay [options] <record-file>\n");
    printf("Options:\n");
    printf(
        "  --convert-json, -cj: Convert the record file to a JSON file in the same directory with record file.\n\
                       When this option is set, it won't replay the record file.\n");
}

Options parseOption(int argc, char* argv[])
{
    Options option;
    char const* arg{};
    if (argc <= 1)
    {
        printUsage();
        exit(1);
    }

    int argIndex = 1;
    while (argIndex < argc)
    {
        arg = argv[argIndex];

        // For anything not starting with a '-', it is a file name
        if (arg[0] != '-')
        {
            option.recordFileName = arg;
            argIndex++;
        }
        else if ((strcmp("--convert-json", arg) == 0) || (strcmp("-cj", arg) == 0))
        {
            option.convertToJson = true;
            argIndex++;
        }
        else if ((strcmp("--help", arg) == 0) || (strcmp("-h", arg) == 0))
        {
            printUsage();
            exit(0);
        }
        else
        {
            // Unknown option
            printf("Unknown option: %s\n", arg);
            printUsage();
            exit(1);
        }
    }

    if (option.recordFileName.getLength() == 0)
    {
        printUsage();
        exit(1);
    }

    return option;
}

int main(int argc, char* argv[])
{
    Options options = parseOption(argc, argv);

    SlangRecord::RecordFileProcessor recordFileProcessor(options.recordFileName);

    Slang::String jsonPath = Slang::Path::replaceExt(options.recordFileName, "json");
    Slang::RefPtr<SlangRecord::JsonConsumer> jsonConsumer;
    SlangRecord::ReplayConsumer replayConsumer;

    SlangRecord::SlangDecoder decoder;

    if (options.convertToJson)
    {
        jsonConsumer = new SlangRecord::JsonConsumer(jsonPath);
        decoder.addConsumer(jsonConsumer.get());
    }
    else
    {
        decoder.addConsumer(&replayConsumer);
    }

    recordFileProcessor.addDecoder(&decoder);

    while (true)
    {
        if (!recordFileProcessor.processNextBlock())
        {
            break;
        }
    }
    return 0;
}
