#ifndef SLANG_ARTIFACT_OUTPUT_UTIL_H
#define SLANG_ARTIFACT_OUTPUT_UTIL_H

#include "../compiler-core/slang-artifact.h"
#include "../compiler-core/slang-diagnostic-sink.h"
#include "../core/slang-basic.h"
#include "slang-com-ptr.h"

namespace Slang
{

class Session;

struct ArtifactOutputUtil
{
    /// Attempts to disassembly artifact into outArtifact.
    /// Errors are output to sink if set. If not desired pass nullptr
    static SlangResult dissassembleWithDownstream(
        Session* session,
        IArtifact* artifact,
        DiagnosticSink* sink,
        IArtifact** outArtifact);

    /// Disassembles if that is plausible
    /// Errors are output to sink if set. If not desired pass nullptr
    static SlangResult maybeDisassemble(
        Session* session,
        IArtifact* artifact,
        DiagnosticSink* sink,
        ComPtr<IArtifact>& outArtifact);

    /// Writes output to writer, will convert into disassembly if that is possible and appropriate
    /// (if outputting to console for example). Errors are output to sink if set. If not desired
    /// pass nullptr
    static SlangResult maybeConvertAndWrite(
        Session* session,
        IArtifact* artifact,
        DiagnosticSink* sink,
        const UnownedStringSlice& writerName,
        ISlangWriter* writer);

    /// Write (without any diagnostics)
    static SlangResult write(IArtifact* artifact, ISlangWriter* writer);
    static SlangResult write(const ArtifactDesc& desc, ISlangBlob* blob, ISlangWriter* writer);

    /// Writes the artifact with diagnostics
    static SlangResult write(
        IArtifact* artifact,
        DiagnosticSink* sink,
        const UnownedStringSlice& writerName,
        ISlangWriter* writer);

    /// Write to the specified path
    static SlangResult writeToFile(
        const ArtifactDesc& desc,
        const void* data,
        size_t size,
        const String& path);
    static SlangResult writeToFile(const ArtifactDesc& desc, ISlangBlob* blob, const String& path);
    static SlangResult writeToFile(IArtifact* artifact, const String& path);
    static SlangResult writeToFile(IArtifact* artifact, DiagnosticSink* sink, const String& path);
};

} // namespace Slang

#endif
