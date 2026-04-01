//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;

[Differentiable]
float test(float x, no_diff out float y)
{
    if (x == 1.0)
        y = 0.0;
    return x * x;
}

[Differentiable]
float caller(float x, no_diff out float y)
{
    return test(x, y);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    var p = diffPair(3.0, 0.0);
    bwd_diff(caller)(p, 1.0);
    outputBuffer[dispatchThreadID.x] = p.d;
    // CHECK: 6.0
}
