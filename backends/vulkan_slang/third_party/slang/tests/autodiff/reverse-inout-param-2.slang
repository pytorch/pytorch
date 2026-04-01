//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct D : IDifferentiable
{
    float m;
    float n;
}

struct ND
{
    float nd;
}

[BackwardDifferentiable]
void g(
    inout no_diff D p,
    out no_diff D po,
    out ND v1,
    inout ND v2,
    float x,
    out float y)
{
    v1 = v2;
    v2.nd = v2.nd + 1.0;
    p.n = v1.nd + 1.0;
    p.m = detach(v2.nd + 1.0 + x); // == v2.nd + 2 + x == 1 + 2 + x == 3+x
    po = p;
    po.m += 1.0; // == 4+x
    y = p.m * x; // == (3+x)*x
}

[BackwardDifferentiable]
void f(inout no_diff D p, out no_diff D p0, out ND v1, inout ND v2, float x, out float y)
{
    // v2.nd is 3.
    g(p, p0, v1, v2, x, y);
    // v2.nd is now 4, now g is equivalent to detach(4+x)*x, so g' = 9.
    g(p, p0, v1, v2, x, y);
}

[ForwardDifferentiable]
float f_ref(float x)
{
    return (3 + 3 * x + x * x) * (3 * x + x * x);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    D p;
    p.m = 1.0;
    p.n = 2.0;

    let v2 : ND = { 1.0 };

    var x = diffPair(5.0);
    float yDiffOut = 1.0;

    __bwd_diff(f)(p, v2, x, yDiffOut);

    outputBuffer[0] = x.p; // should be 5, since bwd_diff does not write back new primal val.
    outputBuffer[1] = x.d; // 9
    outputBuffer[2] = p.m; // 1.0
    outputBuffer[3] = p.n; // 2.0
    outputBuffer[4] = v2.nd; // 1.0
}