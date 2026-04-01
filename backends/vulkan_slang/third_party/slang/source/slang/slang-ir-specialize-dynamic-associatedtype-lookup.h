// slang-ir-specialize-dynamic-associatedtype-lookup.h
#pragma once

namespace Slang
{
struct SharedGenericsLoweringContext;

/// Modifies the lookup of associatedtype entries from witness tables into
/// calls to a specialized "lookup" function that takes a witness table id
/// and returns a witness table id.
/// This is used on GPU targets where all witness tables are replaced as
/// integral IDs instead of a real pointer table.
void specializeDynamicAssociatedTypeLookup(SharedGenericsLoweringContext* sharedContext);

} // namespace Slang
