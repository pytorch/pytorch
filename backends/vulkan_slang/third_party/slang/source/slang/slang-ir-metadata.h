// slang-ir-metadata.h
#pragma once

namespace Slang
{

class ArtifactPostEmitMetadata;
struct IRModule;

void collectMetadata(const IRModule* irModule, ArtifactPostEmitMetadata& outMetadata);

} // namespace Slang
