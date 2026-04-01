//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test to use generics around resource/built in types. 

This test shouldn't compile (wrong type for the Texture1D, but gives this error, which doesn't really explain the problem.

.slang(19): error 30019: expected an expression of type 'Texture1D', got 'Texture1D'
    a.tex = tex;
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

// NOTE! This is purposefully wrong to test tex type checking
Texture1D<float> tex;

struct Another<T>
{
    T z;
    Texture1D<T> tex;
};

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;

    Another<int> a;
    
    a.z = index;
    a.tex = tex;
    
	outputBuffer[index] = a.z + a.tex.Load(index);
}

