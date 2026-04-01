//TEST:SIMPLE(filecheck=CHECK):-emit-spirv-directly -target spirv-asm -entry computeMain -stage compute

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

// CHECK-NOT: ; Annotations
// CHECK: OpCapability Matrix
// CHECK: ; Annotations

//
// This test tests that we can look up unscoped enums based on context, and
// position OpCapability correctly
//
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int i = dispatchThreadID.x;
    int n = outputBuffer[i];
    int r = spirv_asm
    {
        // We should be able to look this up directly, without having to
        // specify CapabilityMatrix or anything
        OpCapability Matrix;
        OpConstant $$int %two 2;
        OpIMul $$int result $n %two
    };
    outputBuffer[i] = r;
}
