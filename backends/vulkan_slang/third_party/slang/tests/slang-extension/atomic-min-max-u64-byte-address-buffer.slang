// No atomic support on CPU
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
// No support for int64_t on DX11
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
// No support for int64_t on fxc - we need SM6.0 and dxil
// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -nvapi-slot u0 -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile cs_6_0 -use-dxil -render-features atomic-int64 -compile-arg -O2 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -render-features atomic-int64 -shaderobj
// For some reason this doesn't work correctly on CUDA? That it behaves as if always does Min. Min and Max do appropriate 
// things tho, because if I force the condition I do get the right answer
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj

// The test doesn't directly use this, but having this defined makes the 0 slot available if NVAPI is going to be used
// Only strictly necessary on the D3D12 path
//TEST_INPUT:ubuffer(data=[0 0 0 0 ], stride=4):name=nvapiBuffer
RWStructuredBuffer<int> nvapiBuffer;

//TEST_INPUT:ubuffer(data=[2 2 2 2 2 2 2 2], stride=4):out,name=outputBuffer
RWByteAddressBuffer outputBuffer;

//TEST_INPUT:ubuffer(data=[0 1 2 3 4 5 6 7], stride=4):name=inputBuffer
RWStructuredBuffer<uint> inputBuffer;

[numthreads(16, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{    
    int tid = dispatchThreadID.x;

    // Produces a different result on CUDA?
   
    uint64_t value;
    {
        int idx = tid & 3;
        // Do 64 bit load that works on CUDA
        value = (uint64_t(inputBuffer[idx * 2 + 1]) << 32) | inputBuffer[idx * 2];
    }

    {
        int idx = (tid & 3) ^ (tid >> 2);
        if (bool(idx & 1))
        {
            outputBuffer.InterlockedMaxU64((idx << 3), value);
        }
        else
        {
            outputBuffer.InterlockedMinU64((idx << 3), value);
        }
    }
}


