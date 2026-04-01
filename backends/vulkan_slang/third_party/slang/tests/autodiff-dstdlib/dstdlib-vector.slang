//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;

[ForwardDifferentiable]
float f(float x)
{
    float3 vx = float3(x, 2*x, 3*x);
    float3 vexpx = exp(vx);
    return vexpx.x + vexpx.y + vexpx.z;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(2.0, 1.0);

        outputBuffer[0] = f(dpa.p);        // Expect: 465.415999
        outputBuffer[1] = __fwd_diff(f)(dpa).d; // Expect: 1326.871736
    }
}