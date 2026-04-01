//TEST:SIMPLE(filecheck=CHECK):-emit-spirv-directly -target spirv-asm -entry computeMain -stage compute

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

// CHECK-NOT: ; Annotations
// CHECK: OpExtension "SPV_KHR_vulkan_memory_model"
// CHECK: ; Annotations

//
// This test tests that we can position OpExtension correctly
//
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int i = dispatchThreadID.x;
    int n = outputBuffer[i];
    int r = spirv_asm
    {
        OpExtension "SPV_KHR_vulkan_memory_model";
        OpConstant $$int %two 2;
        OpIMul $$int result $n %two
    };
    outputBuffer[i] = r;
}
