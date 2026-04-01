//TEST:SIMPLE(filecheck=CHECK): -target spirv

// Check that we don't emit duplicate Flat decorations for the same variable.

// CHECK: OpDecorate %vo_a Flat
// CHECK-NOT: OpDecorate %vo_a Flat

struct VertexOutput
{
    nointerpolation int a : SOME_VALUE;
    float4              b : SV_Position;
};

[shader("vertex")]
VertexOutput Vertex()
{
    VertexOutput out;
    out.a = 0;
    out.b = float4(0, 0, 0, 1);
    return out;
}

[shader("fragment")]
float4 Fragment(in VertexOutput vo)
{
    return float4(float(vo.a), 0, 0, 1);
}