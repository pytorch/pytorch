//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float3> dpfloat3;
typedef DifferentialPair<float3x3> dpmat3;

[BackwardDifferentiable]
float3 diffMul(float3x3 m, float3 v)
{
    return mul(m, v);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    dpfloat3 dpv = dpfloat3(float3(0.5, 1.2, -0.8), float3(0.0, 0.0, 0.0));
    dpmat3 dpm = dpmat3(float3x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
                        float3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    __bwd_diff(diffMul)(dpm, dpv, float3(1.0, 10.0, 100.0));
    outputBuffer[0] = dpm.d[0][0]; // Expect: 0.500000
    outputBuffer[1] = dpm.d[0][1]; // Expect: 1.200000
    outputBuffer[2] = dpm.d[0][2]; // Expect: -0.800000
    outputBuffer[3] = dpm.d[1][0]; // Expect: 5.000000
    outputBuffer[4] = dpm.d[1][1]; // Expect: 12.000000
    outputBuffer[5] = dpm.d[1][2]; // Expect: -8.000000
    outputBuffer[6] = dpm.d[2][0]; // Expect: 50.000000
    outputBuffer[7] = dpm.d[2][1]; // Expect: 120.000000
    outputBuffer[8] = dpm.d[2][2]; // Expect: -80.000000
    outputBuffer[9] = dpv.d[0]; // Expect: 741.000000
    outputBuffer[10] = dpv.d[1]; // Expect: 852.000000
    outputBuffer[11] = dpv.d[2]; // Expect: 963.000000
}
