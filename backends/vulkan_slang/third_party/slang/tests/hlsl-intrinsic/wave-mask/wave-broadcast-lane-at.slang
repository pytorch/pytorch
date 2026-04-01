//TEST_CATEGORY(wave-mask, compute)
//DISABLE_TEST:COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//DISABLE_TEST:COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST:COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_0 -shaderobj
// Disabled on VK because glsl can't do WaveReadLaneAt on matrix. 
//DISABLE_TEST(vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST:COMPARE_COMPUTE_EX:-cuda -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);
    
    const WaveMask mask = 0xf;
    
    int value = 0;
    
    // Scalar
    
    value += WaveMaskBroadcastLaneAt(mask, idx, 1);
    
    // vector
    
    {
        float2 v = float2(idx + 1, idx + 2);
        float2 readValue = WaveMaskBroadcastLaneAt(mask, v, 2);
        
        value += int(readValue[0] + readValue[1]);
    }
        
    // matrix
    {
        matrix<int, 2, 2> v = matrix<int, 2, 2>(idx, idx - 1, idx * 3, idx - 2);
        
        matrix<int, 2, 2> readValue = WaveMaskBroadcastLaneAt(mask, v, 3);
        
        value += int(readValue[0][0] + readValue[0][1] + readValue[1][0] + readValue[1][1]);
    }
    
    outputBuffer[idx] = value;
}