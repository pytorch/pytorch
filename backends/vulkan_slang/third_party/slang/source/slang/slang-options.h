// slang-options.h
#ifndef SLANG_OPTIONS_H
#define SLANG_OPTIONS_H

#include "../compiler-core/slang-source-loc.h"
#include "../core/slang-basic.h"

namespace Slang
{

struct CommandOptions;
class DiagnosticSink;
class IArtifact;

UnownedStringSlice getCodeGenTargetName(SlangCompileTarget target);

SlangResult parseOptions(SlangCompileRequest* compileRequestIn, int argc, char const* const* argv);

// Initialize command options. Holds the details how parsing works.
void initCommandOptions(CommandOptions& commandOptions);

enum class Stage : SlangUInt32;

SlangSourceLanguage findSourceLanguageFromPath(const String& path, Stage& outImpliedStage);

SlangResult createArtifactFromReferencedModule(
    String path,
    SourceLoc loc,
    DiagnosticSink* sink,
    IArtifact** outArtifact);

} // namespace Slang
#endif
