// slang-ir-lower-existential.h
#pragma once

namespace Slang
{
struct SharedGenericsLoweringContext;

/// Lower existential types and related instructions to Tuple types.
void lowerExistentials(SharedGenericsLoweringContext* sharedContext);

} // namespace Slang
