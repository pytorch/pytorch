// This is similar to wave-lane-at.slang but tests more limited supported types for vk.
// We have this 'simple' test, because we can't do matrix (or imat) operations on GLSL/Vk target

//TEST_CATEGORY(wave-mask, compute)
//TEST:COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_0 -shaderobj -render-feature hardware-device
//TEST(vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

// `The input lane index must be uniform across the wave.`. 
// The same restriction applies to glsl/SPIR-V 1.5
// So we are going to use the input buffer to achieve this.

//TEST_INPUT:ubuffer(data=[1 2 3 0], stride=4):name inputBuffer
RWStructuredBuffer<int> inputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    WaveMask mask = 0xf;
  
    int idx = int(dispatchThreadID.x);
    
    int value = 0;
    
    for (int i = 0; i < 4; ++i)
    {
        // Scalar
        
        // The landId is 'dynamic' but it also uniform across the wave (as required by spec)
        const int laneId = inputBuffer[i];
        
        value += WaveMaskReadLaneAt(mask, idx, laneId);
        
        // vector
        
        {
            float2 v = float2(idx + 1, idx + 2);
            float2 readValue = WaveMaskReadLaneAt(mask, v, (laneId + 1) & 3);
            
            value += int(readValue[0] + readValue[1]);
        }            
    }
   
    outputBuffer[idx] = value;
}