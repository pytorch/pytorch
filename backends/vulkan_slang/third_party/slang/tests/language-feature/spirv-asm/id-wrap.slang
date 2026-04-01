//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -shaderobj -emit-spirv-directly

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

// CHECK:      2
// CHECK-NEXT: 4
// CHECK-NEXT: 6
// CHECK-NEXT: 8

//
// This test tests that we correctly wrap constants in ids when required and
// can look up names through that wrapped operand kind
//
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int i = dispatchThreadID.x;
    int n = outputBuffer[i];
    int r = spirv_asm
    {
        OpConstant $$int %two 2;
        // A dummy barrier, this tests that we can look up these names without
        // their Scope and MemorySemantics prefixes, and correctly wrap them in
        // OpConstant.
        OpMemoryBarrier Invocation
            // Test that an unqualified enum works
              AcquireRelease
            // Test that a decimal integer literal works
            | 64
            // Test that a hex integer literal works
            | 0x800
            // Test that an explicit scope works
            | MemorySemanticsAtomicCounterMemory;
        OpIMul $$int result $n %two
    };
    outputBuffer[i] = r;
}
