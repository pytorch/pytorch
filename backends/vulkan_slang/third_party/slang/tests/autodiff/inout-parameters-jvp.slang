//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;

[ForwardDifferentiable]
void g(float x, float y, inout float z)
{
    float m = x + y;
    float n = x - y;
    z += m * n + 2 * x * y;
}

[ForwardDifferentiable]
void h(float x, float y, inout float z)
{
    float m = x + y;
    float n = x - y;
    z = z + m * n + 2 * x * y;
}


[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float x = 2.0;
    float y = 3.5;
    float z = 1.0;
    float dx = 1.0;
    float dy = 0.5;
    float dz = 2.5;

    dpfloat dpz = dpfloat(z, dz);

    __fwd_diff(h)(dpfloat(x, dx), dpfloat(y, dy), dpz);

    outputBuffer[0] = dpz.d; // Expect: 12.0
    outputBuffer[1] = dpz.p; // Expect: 6.75

    __fwd_diff(g)(dpfloat(x, dx), dpfloat(y, dy), dpz);

    outputBuffer[2] = dpz.d; // Expect: 21.5
    outputBuffer[3] = dpz.p; // Expect: 12.5

}