//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float3> dpfloat3;
typedef float3.Differential dfloat3;

typedef DifferentialPair<float2> dpfloat2;
typedef float2.Differential dfloat2;

// This test case should hit a lot of conversion insts
// kIROp_MakeVectorFromScalar (both uint and float), 
// kIROp_CastIntToFloat, etc..
// 
[BackwardDifferentiable]
float3 test_uint_offset(float3 x, float3 y)
{
    uint3 u4 = 1;

    uint3 u5 = u4 + 2;

    return x + y + u5;
}


[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat3 dpx = dpfloat3(float3(2.0, 3.0, 4.0), float3(0.0, 0.0, 0.0));
        dpfloat3 dpy = dpfloat3(float3(1.5, 2.5, 3.5), float3(0.0, 0.0, 0.0));

        __bwd_diff(test_uint_offset)(dpx, dpy, dfloat3(1.0, 2.0, 3.0));
        outputBuffer[0] = dpx.d.y; // Expect: 2
        outputBuffer[1] = dpy.d.y; // Expect: 2
    }

}