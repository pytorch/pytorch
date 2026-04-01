//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct D : IDifferentiable
{
    float n;
    float m;
}
struct C : IDifferentiable
{
    float3 t;
    float v;
}

struct B : IDifferentiable
{
    float2 f;
    C arr[2];
}

struct A : IDifferentiable
{
    float x;
    float y;
    B fb;
    C aarr[3];
    D dv;
};

[BackwardDifferentiable]
A f(A a, int i)
{
    A aout;
    aout.y = 2 * a.x;
    aout.y = aout.y + 2 * a.x;
    aout.x = aout.y + 5 * a.x;
    aout.aarr[1].t = float3(a.y, 0.0, a.x);
    aout.aarr[1].t = float3(a.y, 1.0, a.x + 1.0);
    D nd = { a.x * 4.0f, 1.0f };
    aout.dv = nd;
    aout.dv.m = aout.dv.n * 0.5f;
    // Test that writes to a potentially dynamic address multiple times
    // is allowed and will propagate the correct derivative.
    aout.fb.arr[i].v = a.x * 2.0; // since this value is overwritten, the diff will not accumulate to a.x
    aout.fb.arr[i].v = a.x * 3.0;
    return aout;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    A a = {1.0, 2.0};

    var dpa = diffPair(a);

    A.Differential dout = { 1.0, 1.0, { float2(0), { { float3(1.0), 1.0 }, { float3(1.0), 1.0 } } }, { { float3(1.0), 1.0 }, { float3(1.0), 1.0 }, { float3(1.0), 1.0 } }, {1.0, 1.0} };
    
    __bwd_diff(f)(dpa, 1, dout);
    outputBuffer[0] = dpa.d.x; // Expect: 23
    outputBuffer[1] = dpa.d.y; // Expect: 0
}
