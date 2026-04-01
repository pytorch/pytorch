//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test of a generic function returning a generic struct. 

Works, and deduces the input type of broadcast.
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

vector<T, 4> broadcast4<T>(T v)
{
	vector<T, 4> vec;
 	vec.x = vec.y = vec.z = vec.w = v;
    return vec;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;

    let vec = broadcast4(2.0f);
    
	outputBuffer[index] = dot(vec, vec);
}
