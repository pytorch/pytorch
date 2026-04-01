// Here we are going to do a very simple calculation with double. 
// For simplicity we are not going to use any intrinsics, and not use any resources that use double

//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -render-feature double
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile sm_6_0 -use-dxil -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -render-feature double
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl
// Not supported in WGSL: Double and other unsupported scalar types
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-wgpu

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);
    
    double f = idx * (1.0l / (4.0l));

    f = 1.0l / f;

    {
        uint low, high;
        asuint(f * 2.0l, low, high);
        outputBuffer[idx * 2 + 0] = low;
        outputBuffer[idx * 2 + 1] = high;
    }
}
