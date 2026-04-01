//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;

interface IFoo
{
    // Since IFoo is not inheriting from IDifferentiable,
    // The `this` parameter should be considered as `no_diff` when `getVal`
    // is called through this interface.
    [ForwardDifferentiable]
    float getVal(float y);
}

struct A : IDifferentiable, IFoo
{
    float x;

    // This `getVal` implementation will have `this` parameter treated as
    // differentiable. In order for this method to satisfy the `IFoo.getVal`
    // requirement, we need to synthesize a method with `[NoDiffThis]` attribute
    // that calls this.
    [ForwardDifferentiable]
    float getVal(float y){ return x * x + y * y; }
}

[ForwardDifferentiable]
static float f<T:IFoo>(T obj, float y)
{
    return obj.getVal(y);
}

[ForwardDifferentiable]
static float f2(IFoo obj, float y)
{
    return obj.getVal(y);
}

[ForwardDifferentiable]
float f3(A obj, float y)
{
    return obj.getVal(y);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    A a;
    a.x = 2.0;
    A.Differential ad;
    ad.x = 1.0;

    let rs = __fwd_diff(f)(a, dpfloat(3.0, 1.0));
    outputBuffer[0] = rs.p; // Expect: 13.0
    outputBuffer[1] = rs.d; // Expect: 6.0

    let rs2 = __fwd_diff(f2)(a, dpfloat(3.0, 1.0));
    outputBuffer[2] = rs2.p; // Expect: 13.0
    outputBuffer[3] = rs2.d; // Expect: 6.0

    // By calling A.getVal directly, we will invoke the implementation
    // that differentiates the `this` argument.
    let rs3 = __fwd_diff(f3)(DifferentialPair<A>(a, ad), dpfloat(3.0, 1.0));
    outputBuffer[4] = rs3.p; // Expect: 13.0
    outputBuffer[5] = rs3.d; // Expect: 10.0
}
