#ifndef SLANG_COMPILER_CORE_JSON_SOURCE_MAP_UTIL_H
#define SLANG_COMPILER_CORE_JSON_SOURCE_MAP_UTIL_H

#include "slang-json-value.h"
#include "slang-source-map.h"

namespace Slang
{

struct JSONSourceMapUtil
{
    /// Decode from root into the source map
    static SlangResult decode(
        JSONContainer* container,
        JSONValue root,
        DiagnosticSink* sink,
        SourceMap& out);

    /// Converts the source map contents into JSON
    static SlangResult encode(
        const SourceMap& sourceMap,
        JSONContainer* container,
        DiagnosticSink* sink,
        JSONValue& outValue);

    /// Read the blob (encoded as JSON) as a source map.
    /// Sink is optional, and can be passed as nullptr
    static SlangResult read(ISlangBlob* blob, DiagnosticSink* sink, SourceMap& outSourceMap);
    static SlangResult read(ISlangBlob* blob, SourceMap& outSourceMap);

    /// Write source map to outBlob JSON
    static SlangResult write(const SourceMap& sourceMap, ComPtr<ISlangBlob>& outBlob);
    /// Write out the source map into a blob
    static SlangResult write(
        const SourceMap& sourceMap,
        DiagnosticSink* sink,
        ComPtr<ISlangBlob>& outBlob);
};

} // namespace Slang

#endif // SLANG_COMPILER_CORE_JSON_SOURCE_MAP_UTIL_H
