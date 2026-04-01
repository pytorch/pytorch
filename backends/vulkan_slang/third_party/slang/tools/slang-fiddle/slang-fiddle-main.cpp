// slang-fiddle-main.cpp

#include "core/slang-io.h"
#include "slang-fiddle-diagnostics.h"
#include "slang-fiddle-options.h"
#include "slang-fiddle-scrape.h"
#include "slang-fiddle-template.h"

#if 0
#include "compiler-core/slang-doc-extractor.h"
#include "compiler-core/slang-name-convention-util.h"
#include "compiler-core/slang-name.h"
#include "compiler-core/slang-source-loc.h"
#include "core/slang-file-system.h"
#include "core/slang-list.h"
#include "core/slang-secure-crt.h"
#include "core/slang-string-slice-pool.h"
#include "core/slang-string-util.h"
#include "core/slang-string.h"
#include "core/slang-writer.h"
#include "slang-com-helper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif


namespace fiddle
{
using namespace Slang;

class InputFile : public RefObject
{
public:
    String inputFileName;

    RefPtr<SourceUnit> scrapedSourceUnit;
    RefPtr<TextTemplateFile> textTemplateFile;
};

struct App
{
public:
    App(SourceManager& sourceManager, DiagnosticSink& sink, RootNamePool& rootNamePool)
        : sourceManager(sourceManager), sink(sink), rootNamePool(rootNamePool)
    {
    }

    RootNamePool& rootNamePool;
    SourceManager& sourceManager;
    DiagnosticSink& sink;

    Options options;

    List<RefPtr<InputFile>> inputFiles;
    RefPtr<LogicalModule> logicalModule;

    RefPtr<SourceUnit> parseSourceUnit(SourceView* inputSourceView, String outputFileName)
    {
        return fiddle::parseSourceUnit(
            inputSourceView,
            logicalModule,
            &rootNamePool,
            &sink,
            &sourceManager,
            outputFileName);
    }

    RefPtr<TextTemplateFile> parseTextTemplate(SourceView* inputSourceView)
    {
        return fiddle::parseTextTemplateFile(inputSourceView, &sink);
    }

    String getOutputFileName(String inputFileName) { return inputFileName + ".fiddle"; }

    void processInputFile(String const& inputFileName)
    {
        // The full path to the input and output is determined by the prefixes that
        // were specified via command-line arguments.
        //
        String inputPath = options.inputPathPrefix + inputFileName;

        // We read the fill text of the file into memory as a single string,
        // so that we can easily parse it without need for I/O operations
        // along the way.
        //
        String inputText;
        if (SLANG_FAILED(File::readAllText(inputPath, inputText)))
        {
            sink.diagnose(SourceLoc(), fiddle::Diagnostics::couldNotReadInputFile, inputPath);
            return;
        }

        // Registering the input file with the `sourceManager` allows us
        // to get proper source locations for offsets within it.
        //
        PathInfo inputPathInfo = PathInfo::makeFromString(inputPath);
        SourceFile* inputSourceFile =
            sourceManager.createSourceFileWithString(inputPathInfo, inputText);
        SourceView* inputSourceView =
            sourceManager.createSourceView(inputSourceFile, nullptr, SourceLoc());

        auto inputFile = RefPtr(new InputFile());
        inputFile->inputFileName = inputFileName;

        // We are going to process the same input file in two different ways:
        //
        // - We will read the file using the C++-friendly `Lexer` type
        //   from the Slang `compiler-core` library, in order to scrape
        //   specially marked C++ declarations and process their contents.
        //
        // - We will also read the file as plain text, in order to find
        //   ranges that represent templates to be processed with our
        //   ad hoc Lua-based template engine.

        // We'll do the token-based parsing step first, and allow it
        // to return a `SourceUnit` that we can use to keep track
        // of the file.
        //
        auto sourceUnit = parseSourceUnit(inputSourceView, getOutputFileName(inputFileName));

        // Then we'll read the same file again looking for template
        // lines, and collect that information onto the same
        // object, so that we can emit the file back out again,
        // potentially with some of its content replaced.
        //
        auto textTemplateFile = parseTextTemplate(inputSourceView);

        inputFile->scrapedSourceUnit = sourceUnit;
        inputFile->textTemplateFile = textTemplateFile;

        inputFiles.add(inputFile);
    }

    /// Generate a slug version of the given string.
    ///
    /// A *slug* is a version of a string that has
    /// a visible and obvious dependency on the input
    /// text, but that is massaged to conform to the
    /// constraints of names for some purpose.
    ///
    /// In our case, the constraints are to have an
    /// identifier that is suitable for use as a
    /// preprocessor macro.
    ///
    String generateSlug(String const& inputText)
    {
        StringBuilder builder;
        int prev = -1;
        for (auto c : inputText)
        {
            // Ordinary alphabetic characters go
            // through as-is, but converted to
            // upper-case.
            //
            if (('A' <= c) && (c <= 'Z'))
            {
                builder.appendChar(c);
            }
            else if (('a' <= c) && (c <= 'z'))
            {
                builder.appendChar((c - 'a') + 'A');
            }
            else if (('0' <= c) && (c <= '9'))
            {
                // A digit can be passed through as-is,
                // except that we need to account for
                // the case where (somehow) the very
                // first character is a digit.
                if (prev == -1)
                    builder.appendChar('_');
                builder.appendChar(c);
            }
            else
            {
                // We replace any other character with
                // an underscore (`_`), but we make
                // sure to collapse any sequence of
                // consecutive underscores, and to
                // ignore characters at the start of
                // the string that would turn into
                // underscores.
                //
                if (prev == -1)
                    continue;
                if (prev == '_')
                    continue;

                c = '_';
                builder.appendChar(c);
            }

            prev = c;
        }
        return builder.produceString();
    }

    void generateAndEmitFilesForInputFile(InputFile* inputFile)
    {
        // The output file wil name will be the input file
        // name, but with the suffix `.fiddle` appended to it.
        //
        auto inputFileName = inputFile->inputFileName;
        String outputFileName = getOutputFileName(inputFileName);
        String outputFilePath = options.outputPathPrefix + outputFileName;

        String inputFileSlug = generateSlug(inputFileName);

        // We start the generated file with a header to warn
        // people against editing it by hand (not that doing
        // so will prevent by-hand edits, but its one of the
        // few things we can do).
        //
        StringBuilder builder;
        builder.append("// GENERATED CODE; DO NOT EDIT\n");
        builder.append("//\n");

        builder.append("// input file: ");
        builder.append(inputFile->inputFileName);
        builder.append("\n");

        // There are currently two kinds of generated code
        // we need to handle here:
        //
        // - The code that the scraping tool wants to inject
        //   at each of the `FIDDLE(...)` macro invocation
        //   sites.
        //
        // - The code that is generated from each of the
        //   `FIDDLE TEMPLATE` constructs.
        //
        // We will emit both kinds of output to the same
        // file, to keep things easy-ish for the client.

        // The first kind of output is the content for
        // any `FIDDLE(...)` macro invocations.
        //
        if (hasAnyFiddleInvocations(inputFile->scrapedSourceUnit))
        {

            builder.append("\n// BEGIN FIDDLE SCRAPER OUTPUT\n");
            builder.append("#ifndef ");
            builder.append(inputFileSlug);
            builder.append("_INCLUDED\n");
            builder.append("#define ");
            builder.append(inputFileSlug);
            builder.append("_INCLUDED 1\n");
            builder.append("#ifdef FIDDLE\n");
            builder.append("#undef FIDDLE\n");
            builder.append("#undef FIDDLEX\n");
            builder.append("#undef FIDDLEY\n");
            builder.append("#endif\n");
            builder.append("#define FIDDLEY(ARG) FIDDLE_##ARG\n");
            builder.append("#define FIDDLEX(ARG) FIDDLEY(ARG)\n");
            builder.append("#define FIDDLE FIDDLEX(__LINE__)\n");

            emitSourceUnitMacros(
                inputFile->scrapedSourceUnit,
                builder,
                &sink,
                &sourceManager,
                logicalModule);

            builder.append("\n#endif\n");
            builder.append("// END FIDDLE SCRAPER OUTPUT\n");
        }

        if (inputFile->textTemplateFile->textTemplates.getCount() != 0)
        {
            builder.append("\n// BEGIN FIDDLE TEMPLATE OUTPUT:\n");
            builder.append("#ifdef FIDDLE_GENERATED_OUTPUT_ID\n");

            generateTextTemplateOutputs(
                options.inputPathPrefix + inputFileName,
                inputFile->textTemplateFile,
                builder,
                &sink);

            builder.append("#undef FIDDLE_GENERATED_OUTPUT_ID\n");
            builder.append("#endif\n");
            builder.append("// END FIDDLE TEMPLATE OUTPUT\n");
        }

        builder.append("\n// END OF FIDDLE-GENERATED FILE\n");


        {
            String outputFileContent = builder.produceString();

            if (SLANG_FAILED(File::writeAllTextIfChanged(
                    outputFilePath,
                    outputFileContent.getUnownedSlice())))
            {
                sink.diagnose(
                    SourceLoc(),
                    fiddle::Diagnostics::couldNotWriteOutputFile,
                    outputFilePath);
                return;
            }
        }

        // If we successfully wrote the output file and all of
        // its content, it is time to write out new text for
        // the *input* file, based on the template file.
        //
        {
            String newInputFileContent = generateModifiedInputFileForTextTemplates(
                outputFileName,
                inputFile->textTemplateFile,
                &sink);

            String inputFilePath = options.inputPathPrefix + inputFileName;
            if (SLANG_FAILED(File::writeAllTextIfChanged(
                    inputFilePath,
                    newInputFileContent.getUnownedSlice())))
            {
                sink.diagnose(
                    SourceLoc(),
                    fiddle::Diagnostics::couldNotOverwriteInputFile,
                    inputFilePath);
                return;
            }
        }
    }

    void generateAndEmitFiles()
    {
        for (auto inputFile : inputFiles)
            generateAndEmitFilesForInputFile(inputFile);
    }

    void checkModule() { fiddle::checkModule(this->logicalModule, &sink); }

    void execute(int argc, char const* const* argv)
    {
        // We start by parsing any command-line options
        // that were specified.
        //
        options.parse(sink, argc, argv);
        if (sink.getErrorCount())
            return;

        // All of the code that get scraped will be
        // organized into a single logical module,
        // with no regard for what file each
        // declaration came from.
        //
        logicalModule = new LogicalModule();

        // We iterate over the input paths specified on
        // the command line, to read each in and process
        // its text.
        //
        // This step both scans for declarations that
        // are to be scraped, and also reads the any
        // template spans.
        //
        for (auto inputPath : options.inputPaths)
        {
            processInputFile(inputPath);
        }
        if (sink.getErrorCount())
            return;

        // In order to build up the data model of the
        // scraped declarations (such as what inherits
        // from what), we need to perform a minimal
        // amount of semantic checking here.
        //
        checkModule();
        if (sink.getErrorCount())
            return;


        // Before we go actually running any of the scripts
        // that make up the template files, we need to
        // put things into the environment that will allow
        // those scripts to find the things we've scraped...
        //
        registerScrapedStuffWithScript(logicalModule);
        if (sink.getErrorCount())
            return;


        // Once we've processed the data model, we
        // can generate the code that goes into
        // the corresponding output file, as well
        // as process any templates in the input
        // files.
        //
        generateAndEmitFiles();
        if (sink.getErrorCount())
            return;
    }
};
} // namespace fiddle

#define DEBUG_FIDDLE_COMMAND_LINE 0

#if DEBUG_FIDDLE_COMMAND_LINE
#include <Windows.h>
#endif

int main(int argc, char const* const* argv)
{
    using namespace fiddle;
    using namespace Slang;

    ComPtr<ISlangWriter> writer(new FileWriter(stderr, WriterFlag::AutoFlush));

    RootNamePool rootNamePool;

    SourceManager sourceManager;
    sourceManager.initialize(nullptr, nullptr);

    DiagnosticSink sink(&sourceManager, Lexer::sourceLocationLexer);
    sink.writer = writer;

#if DEBUG_FIDDLE_COMMAND_LINE
    fprintf(stderr, "fiddle:");
    for (int i = 1; i < argc; ++i)
    {
        fprintf(stderr, " %s", argv[i]);
    }
    fprintf(stderr, "\n");

    char buffer[1024];
    GetCurrentDirectoryA(sizeof(buffer), buffer);
    fprintf(stderr, "cwd: %s\n", buffer);
    return 1;
#endif

    try
    {
        App app(sourceManager, sink, rootNamePool);
        app.execute(argc, argv);
    }
    catch (...)
    {
        sink.diagnose(SourceLoc(), fiddle::Diagnostics::internalError);
        return 1;
    }
    return 0;
}
