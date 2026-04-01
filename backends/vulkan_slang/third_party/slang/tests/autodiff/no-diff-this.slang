//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;

struct A : IDifferentiable
{
    float x;

    [ForwardDifferentiable]
    float getVal(float y){ return x * x + y * y; }

    [ForwardDifferentiable]
    [NoDiffThis]
    float getVal2(float y) { return x * x + y * y; }

    [ForwardDifferentiable]
    static float f(A obj, float y)
    {
        return obj.getVal(y);
    }

    [ForwardDifferentiable]
    static float f2(A obj, float y)
    {
        return obj.getVal2(y);
    }
}


[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    A a;
    a.x = 2.0;
    A.Differential ad;
    ad.x = 1.0;

    let rs = __fwd_diff(A.f)(DifferentialPair<A>(a, ad), dpfloat(3.0, 1.0));
    outputBuffer[0] = rs.p; // Expect: 13.0
    outputBuffer[1] = rs.d; // Expect: 10.0

    let rs2 = __fwd_diff(A.f2)(DifferentialPair<A>(a, ad), dpfloat(3.0, 1.0));
    outputBuffer[2] = rs2.p; // Expect: 13.0
    outputBuffer[3] = rs2.d; // Expect: 6.0
}
