//TEST_CATEGORY(wave, compute)
//DISABLE_TEST:COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//DISABLE_TEST:COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST:COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_0 -shaderobj -render-feature hardware-device
// Disabled on VK because glsl can't do WaveReadLaneAt on matrix. 
//DISABLE_TEST(vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST:COMPARE_COMPUTE_EX:-cuda -compute -render-features cuda_sm_7_0 -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

// Note from HLSL: `The input lane index must be uniform across the wave.`. 
// The same restriction applies to glsl/SPIR-V 1.5
// So we are going to use the input buffer to achieve this.

//TEST_INPUT:ubuffer(data=[1 2 3 0], stride=4):name inputBuffer
RWStructuredBuffer<int> inputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);
    
    int value = 0;
    
    for (int i = 0; i < 4; ++i)
    {
        // Scalar
        
        // The landId is 'dynamic' but it also uniform across the wave (as required by spec)
        const int laneId = inputBuffer[i];
        
        value += WaveReadLaneAt(idx, laneId);
        
        // vector
        
        {
            float2 v = float2(idx + 1, idx + 2);
            float2 readValue = WaveReadLaneAt(v, (laneId + 1) & 3);
            
            value += int(readValue[0] + readValue[1]);
        }
            
        // matrix
        {
            matrix<int, 2, 2> v = matrix<int, 2, 2>(idx, idx - 1, idx * 3, idx - 2);
            
            matrix<int, 2, 2> readValue = WaveReadLaneAt(v, (laneId - 1) & 3);
            
            value += int(readValue[0][0] + readValue[0][1] + readValue[1][0] + readValue[1][1]);
        }
    }
    
    outputBuffer[idx] = value;
}