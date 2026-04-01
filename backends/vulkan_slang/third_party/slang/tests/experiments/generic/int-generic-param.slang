//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test around integer generic parameters.

Fails with: 

.slang(7): error 30019: expected an expression of type 'int', got 'typeof(N)'
    return N;
    
This is like *fixed-array.slang*. It needs let.    
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

int getN<N : int>()
{
    return N;
}
    
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;

    float4 values = { 1, 2, 3, 4 };
    
    let val = getN<10>();
    
	outputBuffer[index] = val;
}

