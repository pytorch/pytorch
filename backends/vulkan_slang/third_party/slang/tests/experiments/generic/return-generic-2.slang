//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test of a generic function returning a generic struct.  types. 

Failed:
              
.slang(20): note 99999: an internal error threw an exception while working on code near this location
(0): error 99999: Slang compilation aborted due to an exception of class Slang::InternalError: unexpected: ErrorType              
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

// NOTE, that really I'd prefer, but that won't work because 
// the : int doesn't currently allow
// vector<T, SIZE> makeVector4<T, SIZE : int>(T values[SIZE])

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
    
    let vec = makeVector4<float>(values);
    
	outputBuffer[index] = dot(vec, vec);
}
