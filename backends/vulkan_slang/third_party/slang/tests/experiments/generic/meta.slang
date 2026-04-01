//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test meta programing.

Works.
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct GetVector4<T>
{
    typedef vector<T, 4> Value;
};
  
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;

    GetVector4<float>::Value v = { 1, 2, 3, 4 };
    
	outputBuffer[index] = dot(v, v);
}

