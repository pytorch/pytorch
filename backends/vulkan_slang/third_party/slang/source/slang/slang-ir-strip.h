// slang-ir-strip.h
#pragma once

namespace Slang
{
struct IRModule;

struct IRStripOptions
{
    bool shouldStripNameHints = false;
    bool stripSourceLocs = false;
};

/// Strip out instructions that should only be used by the front-end.
void stripFrontEndOnlyInstructions(IRModule* module, IRStripOptions const& options);

/// Strip witness table entries from imported witness tables.
void stripImportedWitnessTable(IRModule* module);

} // namespace Slang
#pragma once
