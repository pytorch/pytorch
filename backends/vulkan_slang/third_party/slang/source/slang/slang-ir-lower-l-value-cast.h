#ifndef SLANG_IR_LOWER_L_VALUE_CAST_H
#define SLANG_IR_LOWER_L_VALUE_CAST_H

// This defines an IR pass that lowers LValue implicit casts. These are typically formed
// when an in/inout paramter is passed a type that doesn't match.
//
// Depending on the target this could produce
//
// * Nothing - some kinds of casts are implicit for some targets such as HLSL on out parameters for
// same sized integer types
// * A reinterpret cast. On targets with pointers, such as C++/CUDA we can fix the problem by just
// casting to the appropriate pointer (for some kinds of conversions)
// * Creating a temporary of the right type and calling the function, and *converting* to the target
// (say an out parameter)
// * Creating a temporary, converting the value into the temporary, calling the function, and
// converting back to the source

namespace Slang
{

struct IRModule;
class TargetProgram;

void lowerLValueCast(TargetProgram* target, IRModule* module);

} // namespace Slang

#endif
