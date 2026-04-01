//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;

interface IFoo : IDifferentiable
{
    [ForwardDifferentiable]
    float getVal();
}

struct A : IFoo
{
    float x;
    [ForwardDifferentiable]
    float getVal(){return x;}
}

[ForwardDifferentiable]
float f(float x, no_diff IFoo y)
{
    return x * x + y.getVal();
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        A a;
        a.x = 2.0;
        let rs = __fwd_diff(f)(dpfloat(1.5, 1.0), a);
        outputBuffer[0] = rs.p; // Expect: 6.25
        outputBuffer[1] = rs.d; // Expect: 3.0
    }
}
