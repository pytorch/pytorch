// slang-ir-restructure-scoping.h
#pragma once
#include "../core/slang-func-ptr.h"

namespace Slang
{

class RegionTree;
struct IRInst;

/// Fix cases where a value might be used in a non-nested region.
///
/// There can be cases where an IR value V in block A is used in
/// some block B, where A dominates B, *but* when we constructed
/// the region tree, the block B is not in a child/descendent
/// region of A's region, so that it won't be visible through the
/// scoping rules of a target language.
///
/// This function detects such cases, and fixes them up by inserting
/// new temporaries into the IR code so that values that need
/// to survive across blocks are communicated through variables
/// declared at a sufficiently broad scope.
///
void fixValueScoping(RegionTree* regionTree, const Func<bool, IRInst*>& shouldAlwaysFoldInst);

} // namespace Slang
