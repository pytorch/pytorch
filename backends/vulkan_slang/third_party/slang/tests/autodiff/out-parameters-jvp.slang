//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;

[ForwardDifferentiable]
void h(float x, float y, out float result)
{
    float m = x + y;
    float n = x - y;
    result = m * n + 2 * x * y;
}


[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float x = 2.0;
    float y = 3.5;
    float dx = 1.0;
    float dy = 0.5;

    dpfloat dresult;
    __fwd_diff(h)(dpfloat(x, dx), dpfloat(y, dy), dresult);

    outputBuffer[0] = dresult.d; // Expect: 9.5

}