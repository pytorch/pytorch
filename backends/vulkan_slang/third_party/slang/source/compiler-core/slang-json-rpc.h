#ifndef SLANG_COMPILER_CORE_JSON_RPC_H
#define SLANG_COMPILER_CORE_JSON_RPC_H

#include "../core/slang-http.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang-json-parser.h"
#include "slang-json-value.h"
#include "slang.h"

namespace Slang
{

/// Struct to hold values associated with JSON-RPC
struct JSONRPC
{
    enum class ErrorCode
    {
        ParseError = -32700,     ///< Invalid JSON was received by the server.
        InvalidRequest = -32600, ///< The JSON sent is not a valid Request object.
        MethodNotFound = -32601, ///< The method does not exist / is not available.
        InvalidParams = -32602,  ///< Invalid method parameter(s).
        InternalError = -32603,  ///< Internal JSON - RPC error.

        ServerImplStart = -32000, ///< Server implementation defined error range
        ServerImplEnd = -32099,
    };

    static bool isIdOk(const JSONValue& value)
    {
        auto kind = value.getKind();
        switch (kind)
        {
        case JSONValue::Kind::Integer:
        case JSONValue::Kind::Invalid:
        case JSONValue::Kind::String:
            {
                return true;
            }
        }
        return false;
    }

    static const UnownedStringSlice jsonRpc;
    static const UnownedStringSlice jsonRpcVersion;
    static const UnownedStringSlice id;
};

struct JSONRPCErrorResponse
{
    struct Error
    {
        bool isValid() const { return code != 0; }

        Int code = 0;               ///< Value from ErrorCode
        UnownedStringSlice message; ///< Error message

        static const StructRttiInfo g_rttiInfo;
    };

    bool isValid() const
    {
        return jsonrpc == JSONRPC::jsonRpcVersion && error.isValid() && JSONRPC::isIdOk(id);
    }

    UnownedStringSlice jsonrpc = JSONRPC::jsonRpcVersion;
    Error error;
    JSONValue data; ///< Optional data describing the errro
    JSONValue id;   ///< Id associated with this request

    static const StructRttiInfo g_rttiInfo;
};

struct JSONRPCCall
{
    bool isValid() const
    {
        return method.getLength() > 0 && jsonrpc == JSONRPC::jsonRpcVersion && JSONRPC::isIdOk(id);
    }

    UnownedStringSlice jsonrpc = JSONRPC::jsonRpcVersion;
    UnownedStringSlice method; ///< The name of the method
    JSONValue params;          ///< Can be invalid/array/object
    JSONValue id;              ///< Id associated with this request

    static const StructRttiInfo g_rttiInfo;
};

struct JSONResultResponse
{
    bool isValid() const { return jsonrpc == JSONRPC::jsonRpcVersion && JSONRPC::isIdOk(id); }

    UnownedStringSlice jsonrpc = JSONRPC::jsonRpcVersion;
    JSONValue result; ///< The result value
    JSONValue id;     ///< Id associated with this request

    static const StructRttiInfo g_rttiInfo;
};

enum class JSONRPCMessageType
{
    Invalid,
    Result,
    Call,
    Error,
    CountOf,
};

/// Send and receive messages as JSON
class JSONRPCUtil
{
public:
    /// Determine the response type
    static JSONRPCMessageType getMessageType(JSONContainer* container, const JSONValue& value);

    /// Parse slice into JSONContainer. outValue is the root of the hierarchy.
    /// NOTE! Uses and *assumes* there is a source manager on the sink. outValue is likely only
    /// usable whilst the sourceManger is in scope The sourceLoc can only be interpretted with the
    /// sourceLoc anyway
    static SlangResult parseJSON(
        const UnownedStringSlice& slice,
        JSONContainer* container,
        DiagnosticSink* sink,
        JSONValue& outValue);

    /// Convert value into out
    static SlangResult convertToNative(
        JSONContainer* container,
        const JSONValue& value,
        DiagnosticSink* sink,
        const RttiInfo* rttiInfo,
        void* out);
    template<typename T>
    static SlangResult convertToNative(
        JSONContainer* container,
        const JSONValue& value,
        DiagnosticSink* sink,
        T& out)
    {
        return convertToNative(container, value, sink, GetRttiInfo<T>::get(), (void*)&out);
    }

    /// Convert to JSON
    static SlangResult convertToJSON(
        const RttiInfo* rttiInfo,
        const void* in,
        DiagnosticSink* sink,
        StringBuilder& out);

    template<typename T>
    static SlangResult convertToJSON(const T* in, DiagnosticSink* sink, StringBuilder& out)
    {
        return convertToJSON(GetRttiInfo<T>::get(), (const void*)in, sink, out);
    }

    /// Get an id directly from root (assumed id: is in root object definition).
    static JSONValue getId(JSONContainer* container, const JSONValue& root);
};

} // namespace Slang

#endif // SLANG_COMPILER_CORE_JSON_RPC_H
