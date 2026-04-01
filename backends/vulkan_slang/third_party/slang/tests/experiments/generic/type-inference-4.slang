//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test of a generic functions with type inference.

Fails in glslang.

glslang:  .slang(13): error :  '[' :  vector index out of range '2'
glslang:  .slang(13): error :  '' : compilation terminated 
glslang: note : ERROR: 2 compilation errors.  No code generated.
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

T getFirst<T, let N : int>(vector<T, N> val)
{
    return val[2];
}
    
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;

    float2 values = { 1, 2 }; 
    
    let val = getFirst(values);
    
	outputBuffer[index] = val;
}