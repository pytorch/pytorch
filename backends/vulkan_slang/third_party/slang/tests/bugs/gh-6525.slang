//TEST:SIMPLE(filecheck=CHECK): -target spirv

//CHECK: OpEntryPoint

// SV_VertexID is a int in the system binding semantic.
// However, declare it as uint should also work with
// the typeAdapter mechanism in Slang compiler.
in uint gl_VertexIndex : SV_VertexID;

static const float2 uvs[] = { float2(0.0, 0.0), float2(0.0, 2.0), float2(2.0, 0.0) };

[shader("vertex")]
float4 vert_main() : SV_Position{
    const float2 uv = uvs[gl_VertexIndex];
    return float4((uv * 2.0 - 1.0), 0.0, 1.0);
}