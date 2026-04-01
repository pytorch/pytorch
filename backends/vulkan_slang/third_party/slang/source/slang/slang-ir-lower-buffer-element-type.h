#ifndef SLANG_IR_LOWER_BUFFER_ELEMENT_TYPE_H
#define SLANG_IR_LOWER_BUFFER_ELEMENT_TYPE_H

namespace Slang
{
struct IRModule;
class TargetProgram;
struct IRTypeLayoutRules;
struct IRType;

struct BufferElementTypeLoweringOptions
{
    bool lowerBufferPointer = false;

    // For WGSL, we can only create arrays that has a stride of 16 bytes for constant buffers.
    bool use16ByteArrayElementForConstantBuffer = false;
};

// For each struct type S used as element type of a ConstantBuffer, ParameterBlock or
// [RW]StructuredBuffer, we create a lowered type L, where matrix types are lowered to arrays of
// vectors based on major-ness, and loads from the buffer are converted to L_to_S(load(buffer)), and
// stores to the buffer are converted to store(buffer, S_to_L(val)). This pass needs to take place
// after type legalization, and before array return type lowering because it may create functions
// that returns array typed values.
//
void lowerBufferElementTypeToStorageType(
    TargetProgram* target,
    IRModule* module,
    BufferElementTypeLoweringOptions options = BufferElementTypeLoweringOptions());


// Returns the type layout rules should be used for a buffer resource type.
IRTypeLayoutRules* getTypeLayoutRuleForBuffer(TargetProgram* target, IRType* bufferType);
} // namespace Slang

#endif
