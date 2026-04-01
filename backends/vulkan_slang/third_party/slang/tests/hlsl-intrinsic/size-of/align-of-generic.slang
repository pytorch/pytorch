
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer

RWStructuredBuffer<int> outputBuffer;

struct Thing<T>
{
    T t;
};


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int idx = asint(dispatchThreadID.x);

    int size = 0;
    
    switch (idx)
    {   
        
        case 0:     size = alignof(Thing<float>); break;
        case 1:     size = alignof(Thing<float3>); break;
        case 2:     size = alignof(Thing<matrix<uint, 3, 2>>); break;
        case 3:     size = alignof(bool); break;
        default:    break; 
    }

    outputBuffer[idx] = size;
}
