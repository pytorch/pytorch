//TEST_CATEGORY(wave-mask, compute)
//TEST:COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_0 -shaderobj
//TEST(vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const WaveMask mask = 0xf;
    
    int idx = int(dispatchThreadID.x);
    
    int value = 0;
    
    // Scalar

    value += WaveMaskBroadcastLaneAt(mask, idx, 1);
    
    // vector
    
    {
        float2 v = float2(idx + 1, idx + 2);
        float2 readValue = WaveMaskBroadcastLaneAt(mask, v, 4 & 3);
        
        value += int(readValue[0] + readValue[1]);
    }            
   
    outputBuffer[idx] = value;
}