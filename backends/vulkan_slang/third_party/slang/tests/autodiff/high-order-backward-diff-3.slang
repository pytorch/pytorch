//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-dx12 -compute -shaderobj -output-using-type

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
    A rs;
    rs.x = 1.0;
    for (int i = 0; i < 2; i++)
        rs.x = rs.x * x.x;
    return rs.x;
}

[BackwardDifferentiable]
float outerF(A x)
{
    A nx;
    nx.x = 1.0;
    for (int i = 0; i < 2; i++)
        nx.x = nx.x * x.x;
    nx.nx = 2;//x.nx;
    return f(nx);
}

[BackwardDifferentiable]
float df(A x)
{
    A.Differential ad;
    ad.x = 0.0;
    var p = diffPair(x, ad);
    __bwd_diff(outerF)(p, 1.0);
    return p.d.x;
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
    outputBuffer[0] = __fwd_diff(df)(p).d;
}
