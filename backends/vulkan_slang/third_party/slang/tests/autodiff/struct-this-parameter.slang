//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct A : IDifferentiable
{
    float data[3];

    [Differentiable]
    __init()
    {
        [ForceUnroll] for (uint i = 0; i < 3; i++) data[i] = 0.f;
    }

    [mutating]
    [Differentiable]
    void write(float value, inout uint offset)
    {
        data[offset++] = value;
    }

    [mutating]
    [Differentiable]
    void write<let N : int>(vector<float, N> value, inout uint offset)
    {
        [ForceUnroll] for (uint i = 0; i < N; i++) write(value[i], offset);
    }
}

[Differentiable]
float3 run(float3 x)
{
    A a = A();
    uint offset = 0;
    a.write(x, offset);
    return float3(a.data[0], a.data[1], a.data[2]);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    DifferentialPair<float3> dp = diffPair(float3(1.f, 2.f, 3.f), float3(0.f));
    float3 dOut = float3(1.f, 0.5f, 0.1f);
    bwd_diff(run)(dp, dOut);
    outputBuffer[0] = dp.d[0];
    outputBuffer[1] = dp.d[1];
    outputBuffer[2] = dp.d[2];
}
