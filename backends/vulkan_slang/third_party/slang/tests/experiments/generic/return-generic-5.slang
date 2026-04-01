//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test of a generic function returning a generic struct (of built in type). 

Note somewhat silly example because could just use float4(2.0f)

Works.
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

vector<T, N> broadcast<T, let N : int>(T v)
{
	vector<T, N> vec;
    for (int i = 0; i < N; ++i)
    {
        vec[i] = v;
    }
 	
    return vec;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;
    let vec = broadcast<float, 4>(2.0f);
	outputBuffer[index] = dot(vec, vec);
}
