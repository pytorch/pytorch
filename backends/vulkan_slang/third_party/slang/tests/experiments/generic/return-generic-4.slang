//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test of a generic function returning a generic struct.  types. 

Was unable to determine T

Failed:

.slang(35): error 39999: could not specialize generic for arguments of type (float[4])
    let vec = makeVector4(values);
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

vector<T> makeVector4<T>(T values[4])
{
   	vector<T, 4> v;
    v.x = values[0];
    v.y = values[1];
    v.z = values[2];
    v.w = values[3];
    return v;
}
    
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;

    float values[] = { 1, 2, 3, 4};
    
    let vec = makeVector4(values);
    
	outputBuffer[index] = dot(vec, vec);
}