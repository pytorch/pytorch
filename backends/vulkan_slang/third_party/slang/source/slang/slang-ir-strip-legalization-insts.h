// slang-ir-strip-legalization-insts.h
#pragma once

namespace Slang
{

struct IRModule;

/// Removes global instructions from the module that are only required for legalization.
/// These instructions are safe to or must be removed for instruction emitting.
///
/// Currently does the following:
/// - Removes specialization dictionaries.
/// - Removes the contents of all witness table instructions.
/// - Removes global param entry point decorations, as they will cause false code-gen circularity
///   alerts during pre-emit actions.
///
void stripLegalizationOnlyInstructions(IRModule* module);

/// Remove [KeepAlive] decorations from witness tables.
void unpinWitnessTables(IRModule* module);

} // namespace Slang
