// atomic-float-byte-address-buffer.slang

//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-dx11 -slang -compute -render-features atomic-float -output-using-type -nvapi-slot u0 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -render-features atomic-float -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-d3d12 -compute -render-features atomic-float -output-using-type -compile-arg -O2 -nvapi-slot u0 -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-d3d12 -compute -use-dxil -render-features atomic-float -output-using-type -compile-arg -O2 -nvapi-slot u0 -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -output-using-type -shaderobj

// The test doesn't directly use this, but having this defined makes the 0 slot available if NVAPI is going to be used
// Only strictly necessary on the D3D11/D3D12 paths
//TEST_INPUT:ubuffer(data=[0 0 0 0 ], stride=4):name=nvapiBuffer
RWStructuredBuffer<int> nvapiBuffer;

//TEST_INPUT:ubuffer(data=[1.0 2.0 3.0 4.0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

//TEST_INPUT:ubuffer(data=[1.0 2.0 3.0 4.0]):name=workBuffer
RWByteAddressBuffer workBuffer;

//TEST_INPUT:ubuffer(data=[0.7 0.5 0.2 0.6], stride=4):name=anotherBuffer
RWStructuredBuffer<float> anotherBuffer;

[numthreads(16, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    int idx = int((tid & 3) ^ (tid >> 2)); 

    //const float delta = anotherBuffer[idx & 3];
    
    float previousValue = 0;
    workBuffer.InterlockedAddF32((idx << 2), 1.0f, previousValue);
    //workBuffer.InterlockedAddF32((idx ^ 2) << 2, 2.0f + delta);
    
    // The sum of values in anotherBuffer should also be added
    //int anotherIdx = tid >> 2;
    //workBuffer.InterlockedAddF32(anotherIdx << 2, delta);
    
    GroupMemoryBarrierWithGroupSync();
    
    if (tid < 4)
    {
        outputBuffer[tid] = asfloat(workBuffer.Load(int(tid << 2)));
    }
}