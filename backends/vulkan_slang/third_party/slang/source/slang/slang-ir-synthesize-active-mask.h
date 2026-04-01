// slang-ir-synthesize-active-mask.h
#pragma once

namespace Slang
{

class Session;
struct IRModule;
class DiagnosticSink;

/// Synthesize values to represent the "active mask" for warp-/wave-level operations.
///
/// This pass will transform all* functions in `module` that make use of the active
/// mask (directly or indirectly) so that they receive an explicit active mask as
/// an input. It will also transform the bodies of those functions to use that input
/// mask, or values derived from it, as the explicit mask for wave intrinsics.
///
/// * Entry point functions will not be transformed to take an explicit mask, and
/// will instead be changed to compute the active mask to use as the first operation
/// in their body.
///
void synthesizeActiveMask(IRModule* module, DiagnosticSink* sink);

} // namespace Slang
