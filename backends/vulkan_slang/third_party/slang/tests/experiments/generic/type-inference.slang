//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test of a generic functions with type inference.

.slang(17): error 39999: could not specialize generic for arguments of type (float[4])
    let val = get0(values);
                  ^
.slang(5): note 39999: see declaration of func get0<T>(T[4]) -> T

 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

T get0<T>(T values[4])
{
    return values[0];
}
    
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;

    float values[] = { 1, 2, 3, 4};
    
    // NOTE if we specify T then this works. Ie
    // let val = get0<float>(values);
    
    let val = get0(values);
    
	outputBuffer[index] = val;
}