// slang-ir-witness-table-wrapper.h
#pragma once

namespace Slang
{
struct SharedGenericsLoweringContext;

/// This pass generates wrapper functions for witness table function entries.
///
/// Enabled for generation of dynamic dispatch code only.
///
/// Functions that are used to satisfy interface requirement have concrete
/// type signatures for `this` and `associatedtype` parameters/return values.
/// However, when they are called from a witness table, the callee only have a
/// raw pointer for this arguments, since the conrete type is not known to the
/// callee. Therefore, we need to generate wrappers for each member function
/// callable through a witness table, so that the wrapper functions take general void*
/// pointer for arguments whose type is unknown at call sites, and convert them
/// to concrete types and calls the actual implementation.
void generateWitnessTableWrapperFunctions(SharedGenericsLoweringContext* sharedContext);

} // namespace Slang
