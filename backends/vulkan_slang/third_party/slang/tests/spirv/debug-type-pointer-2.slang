//TEST:SIMPLE(filecheck=CHECK): -target spirv

// Check that we do not emit forward reference for debug pointer types if there isn't a need to do so.

// CHECK-NOT: OpExtension "SPV_KHR_relaxed_extended_instruction"
struct T
{
    uint inner;
}

struct Problem
{
    int getVal() { return some_bda_ptr->inner;}
    T *some_bda_ptr;
}

uniform Problem probs;

RWStructuredBuffer<float4> outputBuffer;
[shader("compute")]
void main()
{
    outputBuffer[0] = float(probs.getVal());
}