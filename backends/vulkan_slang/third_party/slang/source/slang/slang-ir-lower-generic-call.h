// slang-ir-lower-generic-call.h
#pragma once

namespace Slang
{
struct SharedGenericsLoweringContext;

/// Lower generic and interface-based code to ordinary types and functions using
/// dynamic dispatch mechanisms.
void lowerGenericCalls(SharedGenericsLoweringContext* sharedContext);

} // namespace Slang
