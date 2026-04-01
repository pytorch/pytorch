//TEST:SIMPLE(filecheck=CHECK): -target spirv
//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-via-glsl

// CHECK: OpDecorate %vout_vertexID{{.*}} PerVertexKHR

struct VertexOutput
{
    nointerpolation int vertexID;
}

[shader("fragment")]
float4 fsmain(VertexOutput vout) : SV_Target
{
    return GetAttributeAtVertex(vout.vertexID, 0);
}
