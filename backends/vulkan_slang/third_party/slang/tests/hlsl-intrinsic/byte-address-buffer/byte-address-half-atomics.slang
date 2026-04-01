// byte-address-half-atomics.slang
// test the atomics on half types.

// Disabled because validation layer doesn't like vector atomics, although nv driver does allow it.
//DISABLED_TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -compute -profile cs_6_2 -render-features half -shaderobj -emit-spirv-directly -output-using-type
//TEST:SIMPLE(filecheck=SPIRV):-target spirv -entry computeMain -stage compute -emit-spirv-directly -skip-spirv-validation
//TEST:SIMPLE(filecheck=HLSL):-target hlsl -entry computeMain -profile cs_6_3
//TEST_INPUT:set tmpBuffer = ubuffer(data=[0 0 0 0], stride=4)
RWByteAddressBuffer tmpBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    half originalValue;
    
    // SPIRV: OpAtomicFAddEXT
    // HLSL: NvInterlockedAddFp16x2
    tmpBuffer.InterlockedAddF16(0, 1.0h, originalValue);
    tmpBuffer.InterlockedAddF16(2, 2.0h, originalValue);
    
    half2 v = tmpBuffer.Load<half2>(0);
    
    // CHECK: 1.0
    outputBuffer[0] = v.x;
    // CHECK: 2.0
    outputBuffer[1] = v.y;
    // CHECK: 0.0
    outputBuffer[3] = originalValue;
}
