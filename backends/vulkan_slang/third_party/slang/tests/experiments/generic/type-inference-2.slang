//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test of a generic functions with type inference.

Does work. 
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

T getFirst<T>(vector<T, 4> val)
{
    return val.x;
}
    
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;

    float4 values = { 1, 2, 3, 4 };
    
    let val = getFirst(values);
    
	outputBuffer[index] = val;
}