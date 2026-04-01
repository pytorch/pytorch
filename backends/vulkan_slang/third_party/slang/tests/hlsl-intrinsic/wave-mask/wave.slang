//TEST_CATEGORY(wave-mask, compute)
//DISABLE_TEST:COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//DISABLE_TEST:COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST:COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_0 -shaderobj -render-feature hardware-device
//TEST(vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -render-feature hardware-device
//TEST:COMPARE_COMPUTE_EX:-cuda -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

//TEST_INPUT:ubuffer(data=[3 10 2 -1  4 53 4 6  1 2 3 4  7 5 3 1], stride=4):name inputBuffer
RWStructuredBuffer<int> inputBuffer;

groupshared int sharedMem[32];

int exclusivePrefixSum(WaveMask mask, int index, int waveLaneId, int originalValue, int elementCount)
{
    WaveMask localMask = WaveMaskBallot(mask, index < elementCount);
    
    sharedMem[index] = 0;
    
    if(waveLaneId < elementCount)
    {
        int temp = 0;
        int val = originalValue;        
        
        for(int i = 1; i < elementCount; i += i)
        {
            int temp = WaveMaskShuffle(localMask, val, waveLaneId - i);            
            if(waveLaneId >= i)
            {
                val += temp;
            }
        }        
        
        // Make it an exclusive prefix sum
        val -= originalValue;
        
        // Write to shared memory
        sharedMem[index] = val;
        return val;
    }    
    
    return 0;    
}

// It matters how kernels with WaveMask intrinsics are launched(!). 
// TODO(JS):
// If I launch with an numthreads amount that is not the size of the Wave on the device, then some 
// lanes will not be executing at startup, and the kernel will have to know that is the case. 
// This works currently though because the mask is only used
// on CUDA, and it's Wave size is 32.
[numthreads(32, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Assumes all threads in the Wave are active at start.
    WaveMask waveMask = ~WaveMask(0);
    
    int index = int(dispatchThreadID.x);
    const int waveLaneId = int(WaveGetLaneIndex());
    
    const int value = inputBuffer[index];
    const int elementCount = 9;
    
    exclusivePrefixSum(waveMask, index, waveLaneId, value, elementCount);
    
    // We don't read from any other lane, so we don't actually need any sync
    //WaveMaskSharedSync(waveMask); 
    
    // It returns the result, but we are going to read from shared memory, to check that aspect worked
    int prefixValue = sharedMem[index];
    
    outputBuffer[index] = prefixValue;
}