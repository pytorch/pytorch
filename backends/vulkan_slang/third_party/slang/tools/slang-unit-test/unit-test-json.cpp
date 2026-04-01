
#include "../../source/compiler-core/slang-json-lexer.h"
#include "../../source/compiler-core/slang-json-parser.h"
#include "../../source/compiler-core/slang-json-value.h"
#include "../../source/core/slang-string-escape-util.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

namespace
{ // anonymous

struct Element
{
    JSONTokenType type;
    const char* value;
};

} // namespace

static SlangResult _lex(const char* in, DiagnosticSink* sink, List<JSONToken>& toks)
{
    SourceManager* sourceManager = sink->getSourceManager();

    String contents(in);
    SourceFile* sourceFile =
        sourceManager->createSourceFileWithString(PathInfo::makeUnknown(), contents);
    SourceView* sourceView = sourceManager->createSourceView(sourceFile, nullptr, SourceLoc());

    JSONLexer lexer;

    lexer.init(sourceView, sink);

    while (lexer.peekType() != JSONTokenType::EndOfFile)
    {
        if (lexer.peekType() == JSONTokenType::Invalid)
        {
            toks.add(lexer.peekToken());
            return SLANG_FAIL;
        }

        toks.add(lexer.peekToken());
        lexer.advance();
    }

    toks.add(lexer.peekToken());

    // If we advance from end of file we should still be at EndOfFile
    SLANG_ASSERT(lexer.advance() == JSONTokenType::EndOfFile);

    return SLANG_OK;
}

static SlangResult _parse(const char* in, DiagnosticSink* sink, JSONListener* listener)
{
    SourceManager* sourceManager = sink->getSourceManager();

    String contents(in);
    SourceFile* sourceFile =
        sourceManager->createSourceFileWithString(PathInfo::makeUnknown(), contents);
    SourceView* sourceView = sourceManager->createSourceView(sourceFile, nullptr, SourceLoc());

    JSONLexer lexer;
    lexer.init(sourceView, sink);

    JSONParser parser;
    SLANG_RETURN_ON_FAIL(parser.parse(&lexer, sourceView, listener, sink));
    return SLANG_OK;
}

static bool _areEqual(
    SourceManager* sourceManager,
    const List<JSONToken>& toks,
    const Element* eles,
    Index elesCount)
{
    if (toks.getCount() != elesCount)
    {
        return false;
    }

    SourceView* sourceView = toks.getCount() ? sourceManager->findSourceView(toks[0].loc) : nullptr;
    const char* const content = sourceView ? sourceView->getContent().begin() : nullptr;

    for (Index i = 0; i < toks.getCount(); ++i)
    {
        const JSONToken& tok = toks[i];
        const auto& ele = eles[i];

        if (tok.type != ele.type)
        {
            return false;
        }

        SLANG_ASSERT(sourceView->getRange().contains(tok.loc));

        const char* start = content + sourceView->getRange().getOffset(tok.loc);

        UnownedStringSlice lexeme(start, tok.length);

        if (lexeme != ele.value)
        {
            return false;
        }
    }

    return true;
}

SLANG_UNIT_TEST(json)
{
    SourceManager sourceManager;
    sourceManager.initialize(nullptr, nullptr);
    DiagnosticSink sink(&sourceManager, nullptr);

    {
        const char text[] =
            " { \"Hello\" : [ \"World\", 1, 2.0, -3.0, -435.5345435, 45e-10, 421.00e+20, 17e1] }";

        const Element eles[] = {
            {JSONTokenType::LBrace, "{"},
            {JSONTokenType::StringLiteral, "\"Hello\""},
            {JSONTokenType::Colon, ":"},
            {JSONTokenType::LBracket, "["},
            {JSONTokenType::StringLiteral, "\"World\""},
            {JSONTokenType::Comma, ","},
            {JSONTokenType::IntegerLiteral, "1"},
            {JSONTokenType::Comma, ","},
            {JSONTokenType::FloatLiteral, "2.0"},
            {JSONTokenType::Comma, ","},
            {JSONTokenType::FloatLiteral, "-3.0"},
            {JSONTokenType::Comma, ","},
            {JSONTokenType::FloatLiteral, "-435.5345435"},
            {JSONTokenType::Comma, ","},
            {JSONTokenType::FloatLiteral, "45e-10"},
            {JSONTokenType::Comma, ","},
            {JSONTokenType::FloatLiteral, "421.00e+20"},
            {JSONTokenType::Comma, ","},
            {JSONTokenType::FloatLiteral, "17e1"},
            {JSONTokenType::RBracket, "]"},
            {JSONTokenType::RBrace, "}"},
            {JSONTokenType::EndOfFile, ""},
        };

        List<JSONToken> toks;
        SLANG_CHECK(SLANG_SUCCEEDED(_lex(text, &sink, toks)));

        SLANG_CHECK(_areEqual(&sourceManager, toks, eles, SLANG_COUNT_OF(eles)));
    }

    {
        StringEscapeHandler* handler = StringEscapeUtil::getHandler(StringEscapeUtil::Style::JSON);


        {
            const auto slice = UnownedStringSlice::fromLiteral("\n\r\b\f\t \"\\/ Some text...");

            SLANG_CHECK(handler->isEscapingNeeded(slice));
            SLANG_CHECK(!handler->isEscapingNeeded(UnownedStringSlice::fromLiteral("Hello!")));

            StringBuilder escaped;
            handler->appendEscaped(slice, escaped);

            StringBuilder unescaped;
            handler->appendUnescaped(escaped.getUnownedSlice(), unescaped);

            SLANG_CHECK(unescaped == slice);
        }

        {
            uint32_t v = 0x7f;

            StringBuilder buf;
            while (v < 0x10000)
            {
                char work[10] = "\\u";

                for (Int i = 0; i < 4; ++i)
                {
                    const uint32_t digitValue = (v >> ((3 - i) * 4)) & 0xf;

                    char digitC =
                        (digitValue > 9) ? char(digitValue - 10 + 'a') : char(digitValue + '0');
                    work[i + 2] = digitC;
                }

                buf << UnownedStringSlice(work, 6);

                v += v;
            }

            // Decode it
            StringBuilder unescaped;
            handler->appendUnescaped(buf.getUnownedSlice(), unescaped);

            // Encode it
            StringBuilder escaped;
            handler->appendEscaped(unescaped.getUnownedSlice(), escaped);

            SLANG_CHECK(escaped == buf);
        }
    }

    {
        const char in[] = "{ \"Hello\" : \"Json\", \"!\" : 10, \"array\" : [1, 2, 3.0] }";

        {
            auto style = JSONWriter::IndentationStyle::Allman;

            JSONWriter writer(style);
            _parse(in, &sink, &writer);

            JSONWriter writerCheck(style);
            _parse(writer.getBuilder().getBuffer(), &sink, &writerCheck);

            SLANG_CHECK(writerCheck.getBuilder() == writer.getBuilder());
        }

        {
            auto style = JSONWriter::IndentationStyle::KNR;

            JSONWriter writer(style, 80);
            _parse(in, &sink, &writer);

            JSONWriter writerCheck(style);
            _parse(writer.getBuilder().getBuffer(), &sink, &writerCheck);

            SLANG_CHECK(writerCheck.getBuilder() == writer.getBuilder());
        }

        {
            // Let's parse into a Value
            RefPtr<JSONContainer> container = new JSONContainer(&sourceManager);

            JSONValue value;
            {
                JSONBuilder builder(container);

                SLANG_CHECK(SLANG_SUCCEEDED(_parse(in, &sink, &builder)));
                value = builder.getRootValue();
            }
            // Let's recreate
            JSONValue copy;
            {
                JSONBuilder builder(container);
                container->traverseRecursively(value, &builder);
                copy = builder.getRootValue();
            }

            SLANG_CHECK(container->areEqual(value, copy));
        }
    }

    {
        // Only need a SourceManager if we are going to store lexemes
        RefPtr<JSONContainer> container = new JSONContainer(nullptr);

        {
            List<JSONValue> values;

            for (Int i = 0; i < 100; ++i)
            {

                values.add(JSONValue::makeInt(i));
                values.add(JSONValue::makeFloat(-double(i)));
            }

            JSONValue array = container->createArray(values.getBuffer(), values.getCount());

            auto arrayView = container->getArray(array);

            SLANG_CHECK(arrayView.getCount() == values.getCount());

            // Check the values are the same
            SLANG_CHECK(container->areEqual(
                arrayView.getBuffer(),
                values.getBuffer(),
                arrayView.getCount()));

            {
                JSONWriter writer(JSONWriter::IndentationStyle::KNR, 80);

                container->traverseRecursively(array, &writer);
            }
        }
        {
            JSONValue obj = JSONValue::makeEmptyObject();

            JSONKey key = container->getKey(UnownedStringSlice::fromLiteral("Hello"));

            container->setKeyValue(obj, key, JSONValue::makeNull());
            container->setKeyValue(obj, key, JSONValue::makeInt(10));

            auto objView = container->getObject(obj);

            SLANG_CHECK(objView.getCount() == 1);

            SLANG_CHECK(objView[0].value.asInteger() == 10);
        }
    }

    // Check repeated keys works out
    // Check out comparison works with different key orders
    {
        RefPtr<JSONContainer> container = new JSONContainer(&sourceManager);
        const char aText[] = "{ \"a\" : 10, \"b\" : 20.0, \"a\" : \"Hello\" }";


        JSONBuilder builder(container);
        SLANG_CHECK(SLANG_SUCCEEDED(_parse(aText, &sink, &builder)));
        const JSONValue a = builder.getRootValue();

        builder.reset();

        const char bText[] = "{ \"b\" : 20.0, \"a\" : \"Hello\"}";
        SLANG_CHECK(SLANG_SUCCEEDED(_parse(bText, &sink, &builder)));
        const JSONValue b = builder.getRootValue();

        SLANG_CHECK(container->areEqual(a, b));

        JSONBuilder convertBuilder(container, JSONBuilder::Flag::ConvertLexemes);

        SLANG_CHECK(SLANG_SUCCEEDED(_parse(aText, &sink, &convertBuilder)));
        const JSONValue c = builder.getRootValue();

        SLANG_CHECK(container->areEqual(a, c));
    }

    {
        RefPtr<JSONContainer> container = new JSONContainer(&sourceManager);
        const char aText[] = "{ \"a\" : \"Hi!\", \"b\" : 20.0, \"c\" : \"Hello\", \"d\" : 30, "
                             "\"e\": null, \"f\": true }";

        JSONBuilder builder(container);
        SLANG_CHECK(SLANG_SUCCEEDED(_parse(aText, &sink, &builder)));
        const JSONValue rootValue = builder.getRootValue();

        List<PersistentJSONValue> values;

        for (char c = 'a'; c <= 'f'; c++)
        {
            const char name[] = {c, 0};
            JSONKey key = container->getKey(UnownedStringSlice(name, 1));
            auto value = container->findObjectValue(rootValue, key);

            SLANG_CHECK(value.type != JSONValue::Type::Invalid);

            PersistentJSONValue persistentValue(value, container);
            values.add(persistentValue);

            PersistentJSONValue copyValue(persistentValue);
            PersistentJSONValue assignValue;
            assignValue = persistentValue;

            SLANG_CHECK(copyValue == persistentValue);
            SLANG_CHECK(assignValue == persistentValue);
        }

        // Destroy the container
        container.setNull();

        SLANG_CHECK(values[0].getSlice() == "Hi!");
        SLANG_CHECK(values[1].asFloat() == 20.0f);
        SLANG_CHECK(values[2].getSlice() == "Hello");
        SLANG_CHECK(values[3].asInteger() == 30);
        SLANG_CHECK(values[4].type == JSONValue::Type::Null);
        SLANG_CHECK(values[5].asBool() == true);
    }
}
