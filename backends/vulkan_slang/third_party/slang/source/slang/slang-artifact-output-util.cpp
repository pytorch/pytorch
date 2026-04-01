#include "slang-artifact-output-util.h"

#include "../core/slang-hex-dump-util.h"
#include "../core/slang-io.h"
#include "../core/slang-platform.h"
#include "../core/slang-string-util.h"
#include "../core/slang-type-text-util.h"

#include <chrono>

// Artifact
#include "../compiler-core/slang-artifact-desc-util.h"
#include "../compiler-core/slang-artifact-util.h"
#include "slang-compiler.h"

namespace Slang
{

/* static */ SlangResult ArtifactOutputUtil::dissassembleWithDownstream(
    Session* session,
    IArtifact* artifact,
    DiagnosticSink* sink,
    IArtifact** outArtifact)
{
    auto desc = artifact->getDesc();

    auto assemblyDesc = desc;
    assemblyDesc.kind = ArtifactKind::Assembly;

    // Check it seems like a plausbile disassembly
    if (!ArtifactDescUtil::isDisassembly(desc, assemblyDesc))
    {
        if (sink)
        {
            sink->diagnose(
                SourceLoc(),
                Diagnostics::cannotDisassemble,
                ArtifactDescUtil::getText(desc));
        }
        return SLANG_FAIL;
    }
    // Get the downstream disassembler that can be used for this target
    // TODO(JS):
    // This could perhaps be performed in some other manner if there was more than one way to
    // produce disassembly from a binary.

    const CodeGenTarget target =
        (CodeGenTarget)ArtifactDescUtil::getCompileTargetFromDesc(assemblyDesc);
    if (target == CodeGenTarget::Unknown)
    {
        return SLANG_FAIL;
    }

    auto downstreamCompiler = getDownstreamCompilerRequiredForTarget(target);

    // Get the required downstream compiler
    IDownstreamCompiler* compiler = session->getOrLoadDownstreamCompiler(downstreamCompiler, sink);

    if (!compiler)
    {
        if (sink)
        {
            auto compilerName =
                TypeTextUtil::getPassThroughAsHumanText((SlangPassThrough)downstreamCompiler);
            sink->diagnose(SourceLoc(), Diagnostics::passThroughCompilerNotFound, compilerName);
        }
        return SLANG_FAIL;
    }
    auto downstreamStartTime = std::chrono::high_resolution_clock::now();
    SLANG_RETURN_ON_FAIL(compiler->convert(artifact, assemblyDesc, outArtifact));
    auto downstreamElapsedTime =
        (std::chrono::high_resolution_clock::now() - downstreamStartTime).count() * 0.000000001;
    session->addDownstreamCompileTime(downstreamElapsedTime);
    return SLANG_OK;
}

SlangResult ArtifactOutputUtil::maybeDisassemble(
    Session* session,
    IArtifact* artifact,
    DiagnosticSink* sink,
    ComPtr<IArtifact>& outArtifact)
{
    const auto desc = artifact->getDesc();
    if (ArtifactDescUtil::isText(desc))
    {
        // Nothing to convert
        return SLANG_OK;
    }

    auto toDesc = desc;
    toDesc.kind = ArtifactKind::Assembly;
    // If this likes a playsible disassebly conversion
    if (ArtifactDescUtil::isDisassembly(desc, toDesc))
    {
        ComPtr<IArtifact> disassemblyArtifact;

        if (SLANG_SUCCEEDED(dissassembleWithDownstream(
                session,
                artifact,
                sink,
                disassemblyArtifact.writeRef())))
        {
            // Check it is now text
            SLANG_ASSERT(ArtifactDescUtil::isText(disassemblyArtifact->getDesc()));

            outArtifact.swap(disassemblyArtifact);
            return SLANG_OK;
        }
    }

    return SLANG_OK;
}

/* static */ SlangResult ArtifactOutputUtil::write(
    const ArtifactDesc& desc,
    ISlangBlob* blob,
    ISlangWriter* writer)
{
    // If is text, we can just output
    if (ArtifactDescUtil::isText(desc))
    {
        auto text = StringUtil::getSlice(blob);
        return writer->write(text.begin(), text.getLength());
    }
    else
    {
        if (writer->isConsole())
        {
            // Else just dump as text
            return HexDumpUtil::dumpWithMarkers(
                (const uint8_t*)blob->getBufferPointer(),
                blob->getBufferSize(),
                24,
                writer);
        }
        else
        {
            // Redirecting stdout to a file, so do the usual thing
            writer->setMode(SLANG_WRITER_MODE_BINARY);
            return writer->write((const char*)blob->getBufferPointer(), blob->getBufferSize());
        }
    }
}

/* static */ SlangResult ArtifactOutputUtil::write(IArtifact* artifact, ISlangWriter* writer)
{
    ComPtr<ISlangBlob> blob;
    SLANG_RETURN_ON_FAIL(artifact->loadBlob(ArtifactKeep::No, blob.writeRef()));
    return write(artifact->getDesc(), blob, writer);
}

static SlangResult _requireBlob(
    IArtifact* artifact,
    DiagnosticSink* sink,
    ComPtr<ISlangBlob>& outBlob)
{
    const auto res = artifact->loadBlob(ArtifactKeep::No, outBlob.writeRef());
    if (SLANG_FAILED(res))
    {
        sink->diagnose(SourceLoc(), Diagnostics::cannotAccessAsBlob);
        return res;
    }
    return SLANG_OK;
}

/* static */ SlangResult ArtifactOutputUtil::write(
    IArtifact* artifact,
    DiagnosticSink* sink,
    const UnownedStringSlice& writerName,
    ISlangWriter* writer)
{
    if (sink == nullptr)
    {
        return write(artifact, writer);
    }

    // Make sure we can access as a blob
    ComPtr<ISlangBlob> blob;
    SLANG_RETURN_ON_FAIL(_requireBlob(artifact, sink, blob));

    const auto res = write(artifact->getDesc(), blob, writer);
    if (SLANG_FAILED(res))
    {
        sink->diagnose(SourceLoc(), Diagnostics::cannotWriteOutputFile, writerName);
    }
    return res;
}

/* static */ SlangResult ArtifactOutputUtil::maybeConvertAndWrite(
    Session* session,
    IArtifact* artifact,
    DiagnosticSink* sink,
    const UnownedStringSlice& writerName,
    ISlangWriter* writer)
{
    // If the output is console we will try and turn into disassembly
    if (writer->isConsole())
    {
        ComPtr<IArtifact> disassemblyArtifact;
        maybeDisassemble(session, artifact, sink, disassemblyArtifact);

        if (disassemblyArtifact)
        {
            return write(disassemblyArtifact, sink, writerName, writer);
        }
    }

    return write(artifact, sink, writerName, writer);
}

/* static */ SlangResult ArtifactOutputUtil::writeToFile(
    const ArtifactDesc& desc,
    const void* data,
    size_t size,
    const String& path)
{
    const SlangResult res =
        ArtifactDescUtil::isText(desc)
            ? File::writeAllTextIfChanged(path, UnownedStringSlice((const char*)data, size))
            : File::writeAllBytes(path, data, size);
    if (desc.kind == ArtifactKind::Executable)
    {
        // Ignore any success code here, assume the one from the actual write is more important.
        SLANG_RETURN_ON_FAIL(File::makeExecutable(path));
    }
    return res;
}

/* static */ SlangResult ArtifactOutputUtil::writeToFile(
    const ArtifactDesc& desc,
    ISlangBlob* blob,
    const String& path)
{
    SLANG_RETURN_ON_FAIL(writeToFile(desc, blob->getBufferPointer(), blob->getBufferSize(), path));
    return SLANG_OK;
}

/* static */ SlangResult ArtifactOutputUtil::writeToFile(IArtifact* artifact, const String& path)
{
    // Get the blob
    ComPtr<ISlangBlob> blob;
    SLANG_RETURN_ON_FAIL(artifact->loadBlob(ArtifactKeep::No, blob.writeRef()));
    return writeToFile(artifact->getDesc(), blob, path);
}

/* static */ SlangResult ArtifactOutputUtil::writeToFile(
    IArtifact* artifact,
    DiagnosticSink* sink,
    const String& path)
{
    if (!sink)
    {
        return writeToFile(artifact, path);
    }

    ComPtr<ISlangBlob> blob;
    SLANG_RETURN_ON_FAIL(_requireBlob(artifact, sink, blob));

    const auto res = writeToFile(artifact, path);
    if (SLANG_FAILED(res) && sink)
    {
        sink->diagnose(SourceLoc(), Diagnostics::cannotWriteOutputFile, path);
    }

    return res;
}

} // namespace Slang
