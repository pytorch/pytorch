// slang-ir-lower-generic-type.h
#pragma once

namespace Slang
{
struct SharedGenericsLoweringContext;

/// Lower all references to generic types (ThisType, AssociatedType, etc.) into IRAnyValueType,
/// and existential types into Tuple<AnyValue, WitnessTable(I), Ptr(RTTIType)>.
void lowerGenericType(SharedGenericsLoweringContext* sharedContext);

} // namespace Slang
