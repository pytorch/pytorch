#include "slang-json-rpc.h"

#include "slang-com-helper.h"
#include "slang-json-native.h"

namespace Slang
{

// https://www.jsonrpc.org/specification


/* static */ const UnownedStringSlice JSONRPC::jsonRpc = UnownedStringSlice::fromLiteral("jsonrpc");
/* static */ const UnownedStringSlice JSONRPC::jsonRpcVersion =
    UnownedStringSlice::fromLiteral("2.0");
/* static */ const UnownedStringSlice JSONRPC::id = UnownedStringSlice::fromLiteral("id");

static const auto g_result = UnownedStringSlice::fromLiteral("result");
static const auto g_error = UnownedStringSlice::fromLiteral("error");
static const auto g_method = UnownedStringSlice::fromLiteral("method");

// Add the fields.
// TODO(JS): This is a little verbose, and could be improved on with something like
// * Tool that automatically generated from C++ (say via the C++ extractor)
// * Macro magic to simplify the construction
static const StructRttiInfo _makeJSONRPCErrorResponse_ErrorRtti()
{
    JSONRPCErrorResponse::Error obj;
    StructRttiBuilder builder(&obj, "JSONRPCErrorResponse::Error", nullptr);
    builder.addField("code", &obj.code);
    builder.addField("message", &obj.message);
    return builder.make();
}
/* static */ const StructRttiInfo JSONRPCErrorResponse::Error::g_rttiInfo =
    _makeJSONRPCErrorResponse_ErrorRtti();

static const StructRttiInfo _makeJSONRPCErrorResponseRtti()
{
    JSONRPCErrorResponse obj;
    StructRttiBuilder builder(&obj, "JSONRPCErrorResponse", nullptr);

    builder.addField("jsonrpc", &obj.jsonrpc);
    builder.addField("error", &obj.error);
    builder.addField("data", &obj.data, StructRttiInfo::Flag::Optional);
    builder.addField("id", &obj.id, StructRttiInfo::Flag::Optional);

    return builder.make();
}
/* static */ const StructRttiInfo JSONRPCErrorResponse::g_rttiInfo =
    _makeJSONRPCErrorResponseRtti();

static const StructRttiInfo _makeJSONRPCCallResponseRtti()
{
    JSONRPCCall obj;
    StructRttiBuilder builder(&obj, "JSONRPCCall", nullptr);

    builder.addField("jsonrpc", &obj.jsonrpc);
    builder.addField("method", &obj.method);
    builder.addField("params", &obj.params, StructRttiInfo::Flag::Optional);
    builder.addField("id", &obj.id, StructRttiInfo::Flag::Optional);
    builder.ignoreUnknownFields();

    return builder.make();
}
/* static */ const StructRttiInfo JSONRPCCall::g_rttiInfo = _makeJSONRPCCallResponseRtti();

static const StructRttiInfo _makeJSONResultResponseResponseRtti()
{
    JSONResultResponse obj;
    StructRttiBuilder builder(&obj, "JSONResultResponse", nullptr);

    builder.addField("jsonrpc", &obj.jsonrpc);
    builder.addField("result", &obj.result);
    builder.addField("id", &obj.id, StructRttiInfo::Flag::Optional);

    return builder.make();
}
/* static */ const StructRttiInfo JSONResultResponse::g_rttiInfo =
    _makeJSONResultResponseResponseRtti();

/* static */ JSONRPCMessageType JSONRPCUtil::getMessageType(
    JSONContainer* container,
    const JSONValue& value)
{
    if (value.getKind() == JSONValue::Kind::Object)
    {
        const JSONKey resultKey = container->findKey(g_result);
        const JSONKey errorKey = container->findKey(g_error);
        const JSONKey methodKey = container->findKey(g_method);

        auto pairs = container->getObject(value);

        for (const auto& pair : pairs)
        {
            if (pair.key == resultKey)
            {
                return JSONRPCMessageType::Result;
            }
            else if (pair.key == errorKey)
            {
                return JSONRPCMessageType::Error;
            }
            else if (pair.key == methodKey)
            {
                return JSONRPCMessageType::Call;
            }
        }
    }

    return JSONRPCMessageType::Invalid;
}

/* static */ SlangResult JSONRPCUtil::parseJSON(
    const UnownedStringSlice& slice,
    JSONContainer* container,
    DiagnosticSink* sink,
    JSONValue& outValue)
{
    SourceManager* sourceManager = sink->getSourceManager();

    // Now need to parse as JSON
    String contents(slice);
    SourceFile* sourceFile =
        sourceManager->createSourceFileWithString(PathInfo::makeUnknown(), contents);
    SourceView* sourceView = sourceManager->createSourceView(sourceFile, nullptr, SourceLoc());

    JSONLexer lexer;
    lexer.init(sourceView, sink);

    JSONBuilder builder(container);

    JSONParser parser;
    SLANG_RETURN_ON_FAIL(parser.parse(&lexer, sourceView, &builder, sink));

    outValue = builder.getRootValue();
    return SLANG_OK;
}

/* static */ SlangResult JSONRPCUtil::convertToNative(
    JSONContainer* container,
    const JSONValue& value,
    DiagnosticSink* sink,
    const RttiInfo* rttiInfo,
    void* out)
{
    auto typeMap = JSONNativeUtil::getTypeFuncsMap();

    JSONToNativeConverter converter(container, &typeMap, sink);
    SLANG_RETURN_ON_FAIL(converter.convert(value, rttiInfo, out));
    return SLANG_OK;
}

/* static */ SlangResult JSONRPCUtil::convertToJSON(
    const RttiInfo* rttiInfo,
    const void* in,
    DiagnosticSink* sink,
    StringBuilder& out)
{
    SourceManager* sourceManager = sink->getSourceManager();
    JSONContainer container(sourceManager);

    auto typeMap = JSONNativeUtil::getTypeFuncsMap();

    NativeToJSONConverter converter(&container, &typeMap, sink);

    JSONValue value;
    SLANG_RETURN_ON_FAIL(converter.convert(rttiInfo, in, value));

    // Convert into a string
    JSONWriter writer(JSONWriter::IndentationStyle::Allman);
    container.traverseRecursively(value, &writer);

    out = writer.getBuilder();
    return SLANG_OK;
}

/* static */ JSONValue JSONRPCUtil::getId(JSONContainer* container, const JSONValue& root)
{
    if (root.getKind() == JSONValue::Kind::Object)
    {
        const JSONKey key = container->findKey(JSONRPC::id);

        if (key != JSONKey(0))
        {
            auto obj = container->getObject(root);
            Index index = obj.findFirstIndex(
                [key](const JSONKeyValue& pair) -> bool { return pair.key == key; });

            if (index >= 0)
            {
                return obj[index].value;
            }
        }
    }
    return JSONValue();
}


} // namespace Slang
