//TEST:SIMPLE(filecheck=CHECK): -target spirv

struct VertexOutput
{
    float color;
}

[shader("fragment")]
float4 fsmain(VertexOutput vout) : SV_Target
{
    // CHECK: ([[# @LINE+1]]): error 39027
    return GetAttributeAtVertex(vout.color, 0); // error: color must be decorated with `nointerpolation`.
}
