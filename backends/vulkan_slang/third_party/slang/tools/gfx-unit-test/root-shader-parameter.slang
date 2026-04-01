// root-shader-parameter.slang

// Test use of root shader parameters.
[__AttributeUsage(_AttributeTargets.Var)]
struct rootAttribute {};

struct S1
{
    StructuredBuffer<uint> c0;
    [root] RWStructuredBuffer<uint> c1;
    StructuredBuffer<uint> c2;
}

struct S0
{
    StructuredBuffer<uint> b0;
    [root] StructuredBuffer<uint> b1;
    ParameterBlock<S1> s1;
    ConstantBuffer<S1> s2;
}

ParameterBlock<S0> g;
[root] RWStructuredBuffer<uint> buffer;

[shader("compute")]
[numthreads(1,1,1)]
void computeMain(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID)
{
    buffer[0] = g.b0[0] - g.b1[0] + g.s1.c0[0] - g.s1.c1[0] + g.s1.c2[0] + g.s2.c0[0] - g.s2.c1[0] + g.s2.c2[0];
    // 10-1+2-3+4+5-6+7
}
