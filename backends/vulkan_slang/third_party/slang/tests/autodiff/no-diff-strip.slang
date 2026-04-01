//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

// This test just checks that we compile and run successfully.
// "NDBuffer<float, 2>" & "no_diff NDBuffer<float, 2>" should resolve to the same code-gen type.
// 

struct NDBuffer<T, let N : int>
{
    RWStructuredBuffer<T> buffer;
    int[N] strides;
    int[N] transform;

    T get(int[N] index) { return buffer[index[0]]; }
}

float _read_slice(int2 index, NDBuffer<float, 2> texture) 
{ 
    return texture.get({index.x, index.y}); 
}

[Differentiable]
void _trampoline(no_diff in vector<int,2> index, in no_diff NDBuffer<float, 2> texture, no_diff out float _result)
{
    _result = _read_slice(index, texture);
}


[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    NDBuffer<float, 2> texture;
    texture.buffer = outputBuffer;
    texture.strides = {1, 1};

    float result;
    _trampoline({dispatchThreadID.x, dispatchThreadID.y}, texture, result);
    outputBuffer[dispatchThreadID.x] = result;
}

