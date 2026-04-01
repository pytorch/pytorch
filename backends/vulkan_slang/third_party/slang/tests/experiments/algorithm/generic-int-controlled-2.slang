//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* Test to control altering of an algorithm via generic compile time values.

Works.
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

int reduce<let Clustering : int, let DupTest : int>( int a)
{
    if (Clustering)
    {
        a = a + a;
    }
    if (DupTest)
    {
        a *= a;
    }
    return a;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = dispatchThreadID.x;

    int r = reduce<1, 2>(index);
    
	outputBuffer[index] = r;
}

