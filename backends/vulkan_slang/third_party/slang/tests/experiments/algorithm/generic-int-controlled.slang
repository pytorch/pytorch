//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* Test to control altering of an algorithm via generic compile time values.

slang(28): note 99999: an internal error threw an exception while working on code near this location
(0): error 99999: Slang compilation aborted due to an exception of class Slang::InternalError: unexpected: could not resolve target declaration for call

Seems like it's the default value (=1) that leads to this crash. If we explicitly specify it's ok.
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

int reduce<let Clustering : int, let DupTest : int = 1>( int a)
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

    int r = reduce<1>(index);
    
	outputBuffer[index] = r;
}