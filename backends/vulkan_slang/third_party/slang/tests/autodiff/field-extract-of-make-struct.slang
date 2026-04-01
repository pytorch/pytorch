//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

// This test checks to ensure that expressions of the form FieldExtract(MakeStruct(...))
// are handled correctly in the presence of differentiation. 
// Such expressions are often optimized out, and the non-differentiability of the struct 
// type can be accidentally overlooked.
//

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

struct Data
{
    [PreferRecompute]
    __init(float tin)
    {
        this.t = tin;
    }
    float t;
};

[BackwardDifferentiable]
float test_make_struct(float y)
{
    Data d = { y };
    return d.t;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(1.0, 0.0);

        bwd_diff(test_make_struct)(dpa, 1.0f);
        outputBuffer[0] = dpa.d; // Expect: 0.0
    }
}