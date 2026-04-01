//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;

struct A : IDifferentiable
{
    float p;
    float3 q;
}

[ForwardDifferentiable]
void g(A a, inout A aout)
{
    float t = a.p + a.q.y * a.q.x;
    aout.p = aout.p + t;
    aout.q = aout.q * t;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float p = 1.0;
    float3 q = float3(1.0, 2.0, 3.0);

    float dp = 1.0;
    float3 dq = float3(1.0, 0.5, 0.25);

    DifferentialPair<A> dpa = DifferentialPair<A>({p, q}, {dp, dq});

    __fwd_diff(g)(DifferentialPair<A>( { p, q }, { dp, dq }), dpa);

    outputBuffer[0] = dpa.p.p; // Expect: 4.0
    outputBuffer[1] = dpa.d.q.x; // Expect: 6.5
    outputBuffer[2] = dpa.d.q.y; // Expect: 8.5
    outputBuffer[3] = dpa.d.q.z; // Expect: 11.25

}