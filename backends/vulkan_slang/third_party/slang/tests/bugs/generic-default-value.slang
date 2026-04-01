//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj 
//TEST(compute):COMPARE_COMPUTE_EX:-vk -slang -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

/* Tests purpose is to confirm that use of `= {}` initialization 
works with a generic */

struct Check<T>
{
    // T is not default initialize type, because it's a generic type parameter.
    // Therefore, when we synthesize the contructor, we won't create a default value
    // for it.
    // __init(T v);
    T v;
};

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = int(dispatchThreadID.x);

    Check<float> v = {0};
    
    outputBuffer[index] = index + int(v.v); 
}

