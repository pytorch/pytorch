// slang-json-rpc-connection.cpp
#include "slang-json-rpc-connection.h"

#include "../core/slang-process-util.h"
#include "../core/slang-short-list.h"
#include "../core/slang-string-util.h"
#include "slang-json-native.h"
#include "slang-json-rpc.h"

namespace Slang
{

/// Ctor
JSONRPCConnection::JSONRPCConnection()
    : m_container(nullptr), m_typeMap(JSONNativeUtil::getTypeFuncsMap())
{
}

SlangResult JSONRPCConnection::init(
    HTTPPacketConnection* connection,
    CallStyle defaultCallStyle,
    Process* process)
{
    m_connection = connection;
    m_process = process;

    {
        // If a call style isn't set, use the prefered style
        const CallStyle preferedCallStyle = CallStyle::Array;
        defaultCallStyle =
            (defaultCallStyle == CallStyle::Default) ? preferedCallStyle : defaultCallStyle;
        m_defaultCallStyle = defaultCallStyle;
    }


    m_sourceManager.initialize(nullptr, nullptr);
    m_diagnosticSink.init(&m_sourceManager, &JSONLexer::calcLexemeLocation);
    m_container.setSourceManager(&m_sourceManager);

    return SLANG_OK;
}

SlangResult JSONRPCConnection::initWithStdStreams(CallStyle defaultCallStyle, Process* process)
{
    RefPtr<Stream> stdinStream, stdoutStream;

    Process::getStdStream(StdStreamType::In, stdinStream);
    Process::getStdStream(StdStreamType::Out, stdoutStream);

    RefPtr<BufferedReadStream> readStream(new BufferedReadStream(stdinStream));

    RefPtr<HTTPPacketConnection> connection = new HTTPPacketConnection(readStream, stdoutStream);
    return init(connection, defaultCallStyle, process);
}

void JSONRPCConnection::clearBuffers()
{
    m_sourceManager.reset();
    m_diagnosticSink.reset();
    m_container.reset();
    m_jsonRoot.reset();
}

bool JSONRPCConnection::isActive()
{
    return m_connection->isActive() && (m_process == nullptr || !m_process->isTerminated());
}

JSONValue JSONRPCConnection::getCurrentMessageId()
{
    SLANG_ASSERT(hasMessage());
    return JSONRPCUtil::getId(&m_container, m_jsonRoot);
}

void JSONRPCConnection::disconnect()
{
    if (m_process)
    {
        if (!m_process->isTerminated())
        {
            if (m_connection)
            {
                // Send. If succeeded, wait
                if (SLANG_SUCCEEDED(sendCall(UnownedStringSlice::fromLiteral("quit"))))
                {
                    // Wait for termination
                    m_process->waitForTermination(m_terminationTimeOutInMs);
                }
            }

            if (!m_process->isTerminated())
            {
                // Okay, just try terminating
                m_process->waitForTermination(m_terminationTimeOutInMs);
            }

            // Okay just kill it then
            if (!m_process->isTerminated())
            {
                m_process->kill(-1);
            }
        }
        m_process.setNull();
    }

    m_connection.setNull();
}

SlangResult JSONRPCConnection::sendRPC(const RttiInfo* rttiInfo, const void* data)
{
    auto typeMap = JSONNativeUtil::getTypeFuncsMap();

    // Convert to JSON
    NativeToJSONConverter converter(&m_container, &typeMap, &m_diagnosticSink);
    JSONValue value;

    SLANG_RETURN_ON_FAIL(converter.convert(rttiInfo, data, value));

    // Convert to text
    JSONWriter writer(JSONWriter::IndentationStyle::Allman);

    m_container.traverseRecursively(value, &writer);
    const StringBuilder& builder = writer.getBuilder();
    return m_connection->write(builder.getBuffer(), builder.getLength());
}

SlangResult JSONRPCConnection::sendError(JSONRPC::ErrorCode code, const JSONValue& id)
{
    return sendError(code, m_diagnosticSink.outputBuffer.getUnownedSlice(), id);
}

SlangResult JSONRPCConnection::sendError(
    JSONRPC::ErrorCode errorCode,
    const UnownedStringSlice& msg,
    const JSONValue& id)
{
    JSONRPCErrorResponse errorResponse;
    errorResponse.error.code = Int(errorCode);
    errorResponse.error.message = msg;
    errorResponse.id = id;

    return sendRPC(&errorResponse);
}

SlangResult JSONRPCConnection::checkArrayObjectWrap(
    const JSONValue& srcArgs,
    const RttiInfo* dstArgsRttiInfo,
    void* dstArgs,
    const JSONValue& id)
{
    if (dstArgsRttiInfo->m_kind == RttiInfo::Kind::Struct &&
        srcArgs.getKind() == JSONValue::Kind::Array)
    {
        auto array = m_container.getArray(srcArgs);
        if (array.getCount() == 1)
        {
            return toNativeOrSendError(array[0], dstArgsRttiInfo, dstArgs, id);
        }
        return SLANG_OK;
    }
    else
    {
        return toNativeOrSendError(srcArgs, dstArgsRttiInfo, dstArgs, id);
    }
}

SlangResult JSONRPCConnection::toNativeArgsOrSendError(
    const JSONValue& srcArgs,
    const RttiInfo* dstArgsRttiInfo,
    void* dstArgs,
    const JSONValue& id)
{
    if (dstArgsRttiInfo->m_kind == RttiInfo::Kind::Struct &&
        srcArgs.getKind() == JSONValue::Kind::Array)
    {
        JSONToNativeConverter converter(&m_container, &m_typeMap, &m_diagnosticSink);
        if (SLANG_FAILED(converter.convertArrayToStruct(srcArgs, dstArgsRttiInfo, dstArgs)))
        {
            return sendError(JSONRPC::ErrorCode::InvalidRequest, id);
        }
        return SLANG_OK;
    }
    else
    {
        return toNativeOrSendError(srcArgs, dstArgsRttiInfo, dstArgs, id);
    }
}

SlangResult JSONRPCConnection::toNativeOrSendError(
    const JSONValue& value,
    const RttiInfo* info,
    void* dst,
    const JSONValue& id)
{
    m_diagnosticSink.outputBuffer.clear();

    JSONToNativeConverter converter(&m_container, &m_typeMap, &m_diagnosticSink);

    if (SLANG_FAILED(converter.convert(value, info, dst)))
    {
        return sendError(JSONRPC::ErrorCode::InvalidRequest, id);
    }

    return SLANG_OK;
}

SlangResult JSONRPCConnection::sendCall(const UnownedStringSlice& method, const JSONValue& id)
{
    JSONRPCCall call;
    call.id = id;
    call.method = method;

    SLANG_RETURN_ON_FAIL(sendRPC(&call));
    return SLANG_OK;
}

SlangResult JSONRPCConnection::sendResult(
    const RttiInfo* rttiInfo,
    const void* result,
    const JSONValue& id)
{
    JSONResultResponse response;
    response.id = id;

    NativeToJSONConverter converter(&m_container, &m_typeMap, &m_diagnosticSink);
    SLANG_RETURN_ON_FAIL(converter.convert(rttiInfo, result, response.result));

    // Send the RPC
    SLANG_RETURN_ON_FAIL(sendRPC(&response));
    return SLANG_OK;
}

SlangResult JSONRPCConnection::sendCall(
    const UnownedStringSlice& method,
    const RttiInfo* argsRttiInfo,
    const void* args,
    const JSONValue& id)
{
    return sendCall(m_defaultCallStyle, method, argsRttiInfo, args, id);
}

SlangResult JSONRPCConnection::sendCall(
    CallStyle callStyle,
    const UnownedStringSlice& method,
    const RttiInfo* argsRttiInfo,
    const void* args,
    const JSONValue& id)
{
    JSONRPCCall call;
    call.id = id;
    call.method = method;

    // Set up the converter to now convert the args.
    NativeToJSONConverter converter(&m_container, &m_typeMap, &m_diagnosticSink);

    // If we have a struct *and* call style is 'array', do special handling
    if (argsRttiInfo->m_kind == RttiInfo::Kind::Struct &&
        _getCallStyle(callStyle) == CallStyle::Array)
    {
        // Convert the args/params in the 'array' style
        SLANG_RETURN_ON_FAIL(converter.convertStructToArray(argsRttiInfo, args, call.params));
    }
    else
    {
        // Convert the args/params in the 'object' sytle
        SLANG_RETURN_ON_FAIL(converter.convert(argsRttiInfo, args, call.params));
    }

    // Send the RPC
    SLANG_RETURN_ON_FAIL(sendRPC(&call));
    return SLANG_OK;
}

SlangResult JSONRPCConnection::waitForResult(Int timeOutInMs)
{
    // Invalidate m_jsonRoot before waitForResult, because when waitForResult fail,
    // we don't want to use the result from the previous read.
    m_jsonRoot.reset();

    SLANG_RETURN_ON_FAIL(m_connection->waitForResult(timeOutInMs));
    return tryReadMessage();
}

SlangResult JSONRPCConnection::tryReadMessage()
{
    m_jsonRoot.reset();

    SLANG_RETURN_ON_FAIL(m_connection->update());
    if (!m_connection->hasContent())
    {
        return SLANG_OK;
    }

    auto content = m_connection->getContent();
    UnownedStringSlice slice((const char*)content.begin(), content.getCount());

    clearBuffers();

    {
        const SlangResult res =
            JSONRPCUtil::parseJSON(slice, &m_container, &m_diagnosticSink, m_jsonRoot);

        // Consume that content/packet
        m_connection->consumeContent();
        if (SLANG_FAILED(res))
        {
            // if we can't parse JSON, we return with id of 'null' as per the standard
            return sendError(JSONRPC::ErrorCode::ParseError, JSONValue::makeNull());
        }
    }

    return SLANG_OK;
}

JSONRPCMessageType JSONRPCConnection::getMessageType()
{
    return JSONRPCUtil::getMessageType(&m_container, m_jsonRoot);
}

SlangResult JSONRPCConnection::getMessage(const RttiInfo* rttiInfo, void* out)
{
    if (!hasMessage())
    {
        return SLANG_FAIL;
    }

    m_diagnosticSink.outputBuffer.clear();
    JSONToNativeConverter converter(&m_container, &m_typeMap, &m_diagnosticSink);

    // Get the RPC response
    JSONResultResponse resultResponse;
    SLANG_RETURN_ON_FAIL(converter.convert(m_jsonRoot, &resultResponse));

    // Convert the result in the response
    SLANG_RETURN_ON_FAIL(converter.convert(resultResponse.result, rttiInfo, out));
    return SLANG_OK;
}

SlangResult JSONRPCConnection::getMessageOrSendError(const RttiInfo* rttiInfo, void* out)
{
    if (!hasMessage())
    {
        return SLANG_FAIL;
    }

    const auto res = getMessage(rttiInfo, out);
    if (SLANG_FAILED(res))
    {
        return sendError(JSONRPC::ErrorCode::ParseError, getCurrentMessageId());
    }
    return res;
}

SlangResult JSONRPCConnection::getRPC(const RttiInfo* rttiInfo, void* out)
{
    if (!hasMessage())
    {
        return SLANG_FAIL;
    }

    m_diagnosticSink.outputBuffer.clear();
    JSONToNativeConverter converter(&m_container, &m_typeMap, &m_diagnosticSink);

    // Convert the result in the response
    SLANG_RETURN_ON_FAIL(converter.convert(m_jsonRoot, rttiInfo, out));
    return SLANG_OK;
}

SlangResult JSONRPCConnection::getRPCOrSendError(const RttiInfo* rttiInfo, void* out)
{
    if (!hasMessage())
    {
        return SLANG_FAIL;
    }

    const auto res = getRPC(rttiInfo, out);
    if (SLANG_FAILED(res))
    {
        return sendError(JSONRPC::ErrorCode::ParseError, getCurrentMessageId());
    }
    return res;
}

} // namespace Slang
