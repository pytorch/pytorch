
#include "../../source/compiler-core/slang-json-lexer.h"
#include "../../source/compiler-core/slang-json-native.h"
#include "../../source/compiler-core/slang-json-parser.h"
#include "../../source/compiler-core/slang-json-source-map-util.h"
#include "../../source/compiler-core/slang-json-value.h"
#include "../../source/compiler-core/slang-source-map.h"
#include "../../source/core/slang-rtti-info.h"
#include "../../source/core/slang-string-escape-util.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

static SlangResult _read(
    JSONContainer* container,
    const String& json,
    DiagnosticSink* sink,
    SourceMap& outSourceMap)
{
    auto sourceManager = sink->getSourceManager();

    JSONValue rootValue;
    {
        // Now need to parse as JSON
        SourceFile* sourceFile =
            sourceManager->createSourceFileWithString(PathInfo::makeUnknown(), json);
        SourceView* sourceView = sourceManager->createSourceView(sourceFile, nullptr, SourceLoc());

        JSONLexer lexer;
        lexer.init(sourceView, sink);

        JSONBuilder builder(container);

        JSONParser parser;
        SLANG_RETURN_ON_FAIL(parser.parse(&lexer, sourceView, &builder, sink));

        rootValue = builder.getRootValue();
    }

    SLANG_RETURN_ON_FAIL(JSONSourceMapUtil::decode(container, rootValue, sink, outSourceMap));

    return SLANG_OK;
}

static SlangResult _write(
    JSONContainer* container,
    const SourceMap& sourceMap,
    DiagnosticSink* sink,
    String& out)
{
    // Write it out
    JSONValue jsonValue;

    SLANG_RETURN_ON_FAIL(JSONSourceMapUtil::encode(sourceMap, container, sink, jsonValue));

    // Convert into a string
    JSONWriter writer(JSONWriter::IndentationStyle::Allman);
    container->traverseRecursively(jsonValue, &writer);

    out = writer.getBuilder();
    return SLANG_OK;
}

static SlangResult _check()
{
    SourceManager sourceManager;
    sourceManager.initialize(nullptr, nullptr);
    DiagnosticSink sink(&sourceManager, nullptr);

    const char jsonSource[] = R"(
{
        "version" : 3,
        "file" : "out.js",
        "sourceRoot" : "",
        "sources" : ["foo.js", "bar.js"],
        "sourcesContent" : [null, null],
        "names" : ["src", "maps", "are", "fun"],
        "mappings" : "A,AAAB;;ABCEG;" 
}
)";

    RefPtr<JSONContainer> container = new JSONContainer(&sourceManager);

    SourceMap sourceMap;
    SLANG_RETURN_ON_FAIL(_read(container, jsonSource, &sink, sourceMap));

    String json;
    SLANG_RETURN_ON_FAIL(_write(container, sourceMap, &sink, json));

    SourceMap readSourceMap;
    SLANG_RETURN_ON_FAIL(_read(container, json, &sink, readSourceMap));

    if (readSourceMap != sourceMap)
    {
        return SLANG_FAIL;
    }

    // Lets try copy construction
    {
        SourceMap copy(sourceMap);

        if (copy != sourceMap)
        {
            return SLANG_FAIL;
        }
    }

    // Lets try assignment
    {
        SourceMap assign;
        assign = sourceMap;
        if (assign != sourceMap)
        {
            return SLANG_FAIL;
        }
    }

    return SLANG_OK;
}

SLANG_UNIT_TEST(sourceMap)
{
    SLANG_CHECK(SLANG_SUCCEEDED(_check()));
}
