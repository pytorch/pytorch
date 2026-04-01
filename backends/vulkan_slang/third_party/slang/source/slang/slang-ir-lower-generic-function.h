// slang-ir-lower-generic-function.h
#pragma once

namespace Slang
{
struct SharedGenericsLoweringContext;

/// Lower generic and interface-based code to ordinary types and functions using
/// dynamic dispatch mechanisms.
/// After this pass, generic type parameters will be lowered into `AnyValue` types,
/// and an existential type I in function signatures will be lowered into
/// `Tuple<AnyValue, WintessTable(I), RTTI*>`.
/// Note that this pass mostly deals with function signatures and interface definitions,
/// and does not modify function bodies.
/// All variable declarations and type uses are handled in `lower-generic-type`,
/// and all call sites are handled in `lower-generic-call`.
void lowerGenericFunctions(SharedGenericsLoweringContext* sharedContext);

} // namespace Slang
