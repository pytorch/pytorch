//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef DifferentialPair<float3> dpfloat3;

[ForwardDifferentiable]
float f(float a)
{
    return a * a + a;
}

[ForwardDifferentiable]
float f(float3 a)
{
    return a.x * a.y + a.z;
}

[ForwardDifferentiable]
float g(float a)
{
    // df((2.0, 4.0, 6.0), (1.0, 2.0, 3.0)) 
    // 2.0 * 2.0 + 4.0 * 1.0 + 3.0 = 11.0
    return f(float3(a, 2*a, 3*a));
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(2.0, 1.0);
        dpfloat3 dpf3 = dpfloat3(float3(1.0, 3.0, 5.0), float3(0.5, 1.5, 2.5));

        outputBuffer[0] = f(dpa.p);                               // Expect: 6
        outputBuffer[1] = f(dpf3.p);                              // Expect: 8
        outputBuffer[2] = __fwd_diff(f)(dpf3).d;                       // Expect: 5.5
        outputBuffer[3] = __fwd_diff(f)(dpa).d;                        // Expect: 5
        outputBuffer[4] = __fwd_diff(g)(dpa).d;                        // Expect: 11.0
    }
}
