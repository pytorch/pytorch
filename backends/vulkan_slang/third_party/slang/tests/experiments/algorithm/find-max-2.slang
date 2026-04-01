//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* Find max algorithm. 

Works.
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IComparable
{
    bool isLess(This other);
};

extension int : IComparable
{
	bool isLess(int other) { return this < other; }
};

T findMax<T : IComparable, let SIZE : int>(T arr[SIZE])
{
    T m = arr[0];
    for (int i = 1; i < 8; ++i)
    {
        if (m.isLess(arr[i]))
        {
            m = arr[i];
        }
    }
    return m;
}


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = dispatchThreadID.x;

    int values[] = { 1, 3, - 1};
    
	outputBuffer[index] = findMax<int, 3>(values);
}


