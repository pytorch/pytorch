// perfect-hash-main.cpp

#include "../../source/compiler-core/slang-json-parser.h"
#include "../../source/compiler-core/slang-json-value.h"
#include "../../source/compiler-core/slang-lexer.h"
#include "../../source/compiler-core/slang-perfect-hash-codegen.h"
#include "../../source/core/slang-io.h"
#include "../../source/core/slang-secure-crt.h"
#include "../../source/core/slang-string-util.h"

#include <stdio.h>

using namespace Slang;

static SlangResult parseJson(const char* inputPath, DiagnosticSink* sink, JSONListener& listener)
{
    auto sourceManager = sink->getSourceManager();

    String contents;
    SLANG_RETURN_ON_FAIL(File::readAllText(inputPath, contents));
    PathInfo pathInfo = PathInfo::makeFromString(inputPath);
    SourceFile* sourceFile = sourceManager->createSourceFileWithString(pathInfo, contents);
    SourceView* sourceView = sourceManager->createSourceView(sourceFile, nullptr, SourceLoc());
    JSONLexer lexer;
    lexer.init(sourceView, sink);
    JSONParser parser;
    SLANG_RETURN_ON_FAIL(parser.parse(&lexer, sourceView, &listener, sink));
    return SLANG_OK;
}

// Extract from a json value, the "opname" member from all the objects in the
// "instructions" array.
// Returns the empty list on failure
static List<String> extractOpNames(
    UnownedStringSlice& error,
    const JSONValue& v,
    JSONContainer& container)
{
    List<String> opnames;

    // Wish we could just write à la jq
    // List<String> result = match(myJSONValue, "instructions", AsArray, "opname", AsString);
    const auto instKey = container.findKey(UnownedStringSlice("instructions"));
    const auto opnameKey = container.findKey(UnownedStringSlice("opname"));
    const auto aliasesKey = container.findKey(UnownedStringSlice("aliases"));
    if (!instKey)
    {
        error = UnownedStringSlice("JSON parsing failed, no \"instructions\" key\n");
        return {};
    }
    if (!opnameKey)
    {
        error = UnownedStringSlice("JSON parsing failed, no \"opname\" key\n");
        return {};
    }

    const auto instructions = container.findObjectValue(v, instKey);
    if (!instructions.isValid() || instructions.type != JSONValue::Type::Array)
    {
        error =
            UnownedStringSlice("JSON parsing failed, no \"instructions\" member of array type\n");
        return {};
    }
    for (const auto& inst : container.getArray(instructions))
    {
        const auto opname = container.findObjectValue(inst, opnameKey);
        if (!opname.isValid() || opname.getKind() != JSONValue::Kind::String)
        {
            error = UnownedStringSlice(
                "JSON parsing failed, no \"opname\" member of string type for instruction\n");
            return {};
        }
        opnames.add(container.getString(opname));

        if (aliasesKey)
        {
            auto aliases = container.findObjectValue(inst, aliasesKey);
            if (aliases.isValid() && aliases.type == JSONValue::Type::Array)
            {
                for (auto& alias : container.getArray(aliases))
                {
                    opnames.add(container.getString(alias));
                }
            }
        }
    }

    return opnames;
}

int main(int argc, const char* const* argv)
{
    using namespace Slang;

    if (argc != 6)
    {
        fprintf(
            stderr,
            "Usage: %s input.grammar.json output.cpp enum-name enumerant-prefix enum-header-file\n",
            argc >= 1 ? argv[0] : "slang-lookup-generator");
        return 1;
    }

    const char* const inPath = argv[1];
    const char* const outCppPath = argv[2];
    const char* const enumName = argv[3];
    const char* const enumerantPrefix = argv[4];
    const char* const enumHeader = argv[5];

    RefPtr<FileWriter> writer(new FileWriter(stderr, WriterFlag::AutoFlush));
    SourceManager sourceManager;
    sourceManager.initialize(nullptr, nullptr);
    DiagnosticSink sink(&sourceManager, Lexer::sourceLocationLexer);
    sink.writer = writer;

    List<String> opnames;

    if (String(inPath).endsWith("json"))
    {
        // If source is a json file parse it.
        JSONContainer container(sink.getSourceManager());
        JSONBuilder builder(&container);
        if (SLANG_FAILED(parseJson(inPath, &sink, builder)))
        {
            sink.diagnoseRaw(Severity::Error, "Json parsing failed\n");
            return 1;
        }

        UnownedStringSlice error;
        opnames = extractOpNames(error, builder.getRootValue(), container);
        if (error.getLength())
        {
            sink.diagnoseRaw(Severity::Error, error);
            return 1;
        }
    }
    else
    {
        // Otherwise, we assume the input is a text file with one name per line.
        String content;
        File::readAllText(inPath, content);
        List<UnownedStringSlice> words;
        StringUtil::split(content.getUnownedSlice(), '\n', words);
        for (auto w : words)
            opnames.add(w);
    }

    if (SLANG_FAILED(writePerfectHashLookupCppFile(
            outCppPath,
            opnames,
            enumName,
            enumerantPrefix,
            enumHeader,
            &sink)))
        return -1;

    return 0;
}
