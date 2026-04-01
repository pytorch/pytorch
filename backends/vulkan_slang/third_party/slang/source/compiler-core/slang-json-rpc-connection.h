#ifndef SLANG_COMPILER_CORE_JSON_RPC_CONNECTION_H
#define SLANG_COMPILER_CORE_JSON_RPC_CONNECTION_H

#include "../../source/core/slang-http.h"
#include "../../source/core/slang-process.h"
#include "slang-diagnostic-sink.h"
#include "slang-json-diagnostics.h"
#include "slang-json-rpc.h"
#include "slang-json-value.h"
#include "slang-source-loc.h"
#include "slang-test-server-protocol.h"

namespace Slang
{

/* A type to handle communication via the JSON-RPC protocol.

Uses Rtti to be able to convert between native and JSON types.
Provides methods that work on the JSON-RPC protocol, these methods contain 'RPC' and can only
use the JSON-RPC protocol types. These types will hold items that can vary (like parameters)
in JSONValue parameters. Code can use regular JSON functions to access/process.

Doing conversions to native types and JSON manually can be a fairly monotonous task. To avoid this
effort Rtti and JSON<->Rtti conversions can be used. For example sendCall will send a JSON-RPC
'call' method, with the parameters being converted from some native type. For this to work the type
T must be determinable via GetRttiType<T>, and T must only contain types that JSON<->Rtti conversion
supports.
*/
class JSONRPCConnection : public RefObject
{
public:
    enum class CallStyle
    {
        Default, ///< The default
        Object,  ///< Params are passed as an object
        Array,   ///< Params are passed as an array
    };

    /// An init function must be called before use
    /// If a process is implementing the server it should be passed in if the process needs to shut
    /// down if the connection does
    SlangResult init(
        HTTPPacketConnection* connection,
        CallStyle callStyle = CallStyle::Default,
        Process* process = nullptr);

    /// Initialize using stdin/out streams for input/output.
    SlangResult initWithStdStreams(
        CallStyle callStyle = CallStyle::Default,
        Process* process = nullptr);

    /// Disconnect. May block while server shuts down
    void disconnect();

    SlangResult checkArrayObjectWrap(
        const JSONValue& srcArgs,
        const RttiInfo* dstArgsRttiInfo,
        void* dstArgs,
        const JSONValue& id);

    /// Convert value to dst. Will write response on fails
    SlangResult toNativeOrSendError(
        const JSONValue& value,
        const RttiInfo* info,
        void* dst,
        const JSONValue& id);

    template<typename T>
    SlangResult toNativeOrSendError(const JSONValue& value, T* data, const JSONValue& id)
    {
        return toNativeOrSendError(value, GetRttiInfo<T>::get(), data, id);
    }

    /// Convert value to dst.
    /// The 'Args' aspect here is to handle Args/Params in JSON-RPC which can be specified as an
    /// array or object style. This call will automatically handle either case. toNativeOrSendError
    /// does not assume the thing being converted is args, and so doesn't allow such a
    /// transformation. Will write error response on failure.
    SlangResult toNativeArgsOrSendError(
        const JSONValue& srcArgs,
        const RttiInfo* dstArgsRttiInfo,
        void* dstArgs,
        const JSONValue& id);

    template<typename T>
    SlangResult toNativeArgsOrSendError(const JSONValue& srcArgs, T* dstArgs, const JSONValue& id)
    {
        return toNativeArgsOrSendError(srcArgs, GetRttiInfo<T>::get(), dstArgs, id);
    }

    template<typename T>
    SlangResult toValidNativeOrSendError(const JSONValue& value, T* data, const JSONValue& id);

    /// Send a RPC response (ie should only be one of the JSONRPC classes)
    SlangResult sendRPC(const RttiInfo* info, const void* data);
    template<typename T>
    SlangResult sendRPC(const T* data)
    {
        return sendRPC(GetRttiInfo<T>::get(), (const void*)data);
    }

    /// Send an error
    SlangResult sendError(JSONRPC::ErrorCode code, const JSONValue& id);
    SlangResult sendError(
        JSONRPC::ErrorCode errorCode,
        const UnownedStringSlice& msg,
        const JSONValue& id);

    /// Send a 'call'
    /// Uses the default CallStyle as set when init
    SlangResult sendCall(
        const UnownedStringSlice& method,
        const RttiInfo* argsRttiInfo,
        const void* args,
        const JSONValue& id = JSONValue());
    template<typename T>
    SlangResult sendCall(
        const UnownedStringSlice& method,
        const T* args,
        const JSONValue& id = JSONValue())
    {
        return sendCall(method, GetRttiInfo<T>::get(), (const void*)args, id);
    }

    /// Send a 'call'
    /// Uses the call mechanism specified in callStyle. It is valid to pass as Default.
    SlangResult sendCall(
        CallStyle callStyle,
        const UnownedStringSlice& method,
        const RttiInfo* argsRttiInfo,
        const void* args,
        const JSONValue& id = JSONValue());
    template<typename T>
    SlangResult sendCall(
        CallStyle callStyle,
        const UnownedStringSlice& method,
        const T* args,
        const JSONValue& id = JSONValue())
    {
        return sendCall(callStyle, method, GetRttiInfo<T>::get(), (const void*)args, id);
    }

    /// Send a call, wheret there are no arguments
    SlangResult sendCall(const UnownedStringSlice& method, const JSONValue& id = JSONValue());

    template<typename T>
    SlangResult sendResult(const T* result, const JSONValue& id)
    {
        return sendResult(GetRttiInfo<T>::get(), (const void*)result, id);
    }
    SlangResult sendResult(const RttiInfo* rttiInfo, const void* result, const JSONValue& id);

    /// Try to read a message. Will return if message is not available.
    SlangResult tryReadMessage();

    /// Will block for message/result up to time
    SlangResult waitForResult(Int timeOutInMs = -1);

    /// If we have an JSON-RPC message m_jsonRoot the root.
    bool hasMessage() const { return m_jsonRoot.isValid(); }

    /// If there is a message returns kind of JSON RPC message
    JSONRPCMessageType getMessageType();

    /// Get JSON-RPC message (ie one of JSONRPC classes)
    template<typename T>
    SlangResult getRPC(T* out)
    {
        return getRPC(GetRttiInfo<T>::get(), (void*)out);
    }
    SlangResult getRPC(const RttiInfo* rttiInfo, void* out);

    /// Get JSON-RPC message (ie one of JSONRPC prefixed classes)
    /// If there is a message and there is a failure, will send an error response
    template<typename T>
    SlangResult getRPCOrSendError(T* out)
    {
        return getRPCOrSendError(GetRttiInfo<T>::get(), (void*)out);
    }
    SlangResult getRPCOrSendError(const RttiInfo* rttiInfo, void* out);

    /// Get message (has to be part of JSONRPCResultResponse)
    template<typename T>
    SlangResult getMessage(T* out)
    {
        return getMessage(GetRttiInfo<T>::get(), (void*)out);
    }
    SlangResult getMessage(const RttiInfo* rttiInfo, void* out);

    /// If there is a message and there is a failure, will send an error response
    template<typename T>
    SlangResult getMessageOrSendError(T* out)
    {
        return getMessageOrSendError(GetRttiInfo<T>::get(), (void*)out);
    }
    SlangResult getMessageOrSendError(const RttiInfo* rttiInfo, void* out);

    /// Clears all the internal buffers (for JSON/Source/etc).
    /// Happens automatically on tryReadMessage/readMessage
    void clearBuffers();

    /// True if this connection is active
    bool isActive();

    /// Get the id of the current message
    JSONValue getCurrentMessageId();

    /// Get the diagnostic sink. Can queue up errors before sending an error
    DiagnosticSink* getSink() { return &m_diagnosticSink; }

    /// Get the container
    JSONContainer* getContainer() { return &m_container; }

    /// Turn a value into a persistant value. This will also remove any sourceLoc under the
    /// assumption that it's highly likely it will become invalid in most usage scenarios.
    PersistentJSONValue getPersistentValue(const JSONValue& value)
    {
        return PersistentJSONValue(value, &m_container, SourceLoc());
    }

    HTTPPacketConnection* getUnderlyingConnection() { return m_connection.Ptr(); }

    /// Dtor
    ~JSONRPCConnection() { disconnect(); }

    /// Ctor
    JSONRPCConnection();

protected:
    CallStyle _getCallStyle(CallStyle callStyle) const
    {
        return (callStyle == CallStyle::Default) ? m_defaultCallStyle : callStyle;
    }

    RefPtr<Process> m_process;                 ///< Backing process (optional)
    RefPtr<HTTPPacketConnection> m_connection; ///< The underlying 'transport' connection, whilst
                                               ///< HTTP currently doesn't have to be

    DiagnosticSink m_diagnosticSink; ///< Holds any diagnostics typically generated by parsing JSON,
                                     ///< producing JSON from native types

    SourceManager
        m_sourceManager; ///< Holds the JSON text for current message/output. Is cleared regularly.
    JSONContainer m_container; ///< Holds the backing memory for jsonMemory, and used when
                               ///< converting input into output JSON

    JSONValue m_jsonRoot; ///< The root JSON value for the currently read message.

    CallStyle m_defaultCallStyle = CallStyle::Array; ///< The default calling style

    RttiTypeFuncsMap m_typeMap;

    Int m_terminationTimeOutInMs =
        1 * 1000; ///< Time to wait for termination response. Default is 1 second
};

// ---------------------------------------------------------------------------
template<typename T>
SlangResult JSONRPCConnection::toValidNativeOrSendError(
    const JSONValue& value,
    T* data,
    const JSONValue& id)
{
    const RttiInfo* rttiInfo = GetRttiInfo<T>::get();

    SLANG_RETURN_ON_FAIL(toNativeOrSendError(value, rttiInfo, (void*)data, id));
    if (!data->isValid())
    {
        // If it has a name add validation info
        if (rttiInfo->isNamed())
        {
            const NamedRttiInfo* namedRttiInfo = static_cast<const NamedRttiInfo*>(rttiInfo);
            m_diagnosticSink.diagnose(
                SourceLoc(),
                JSONDiagnostics::argsAreInvalid,
                namedRttiInfo->m_name);
        }

        return sendError(JSONRPC::ErrorCode::InvalidRequest, id);
    }
    return SLANG_OK;
}

} // namespace Slang

#endif // SLANG_COMPILER_CORE_JSON_RPC_CONNECTION_H
