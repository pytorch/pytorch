//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

[BackwardDerivative(bwd_f)]
void f(float x, out float y)
{
    y = x * x;
}

void bwd_f(inout DifferentialPair<float> dpx, in float.Differential dy)
{
    dpx = DifferentialPair<float>(dpx.p, 2.0 * dpx.p * dy);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        float o;
        f(3.0, o);

        dpfloat dpa = dpfloat(2.0, 1.0);
        dpfloat dpo = dpfloat(2.0, 1.0);
        __bwd_diff(f)(dpa, o);

        outputBuffer[0] = o; // Expect: 9
        outputBuffer[1] = dpa.d; // Expect: 36
    }
}
