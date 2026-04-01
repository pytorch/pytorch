//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* Find max algorithm. 

Doesn't work because, can't compare T as it is any type.

.slang(15): error 39999: no overload for '>' applicable to arguments of type (T, T)
        if (arr[i] > m)
                   ^
core.meta.slang(2572): note 39999: candidate: func >(uint64_t, uint64_t) -> bool

 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

T findMax<T, let SIZE : int>(T arr[SIZE])
{
    T m = arr[0];
    for (int i = 1; i < 8; ++i)
    {
        if (arr[i] > m)
        {
            m = arr[i];
        }
    }
    return i;
}


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = dispatchThreadID.x;

    int values[] = { 1, 3, - 1};
    
	outputBuffer[index] = findMax(values);
}


