//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct A : IDifferentiable
{
    float x;
    int nx;
}

[BackwardDifferentiable]
float f(A x)
{
    return x.x * x.x;
}

[BackwardDifferentiable]
float outerF(A x)
{
    A nx;
    nx.x = x.x * x.x;
    nx.nx = 2;//x.nx;
    return f(nx);
}

[BackwardDifferentiable]
float df(A x)
{
    A.Differential ad;
    ad.x = 1.0;
    return __fwd_diff(outerF)(DifferentialPair<A>(x, ad)).d; // 4*x^3
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Given f(x) = x^4,
    // f''(x) = 12 * x^2
    // Expect f''(4) = 192
    A a;
    a.x = 4.0;
    a.nx = 54;
    A.Differential ad;
    ad.x = 1.0;
    var p = diffPair(a, ad);
    __bwd_diff(df)(p, 1.0);
    outputBuffer[0] = p.d.x;
}
