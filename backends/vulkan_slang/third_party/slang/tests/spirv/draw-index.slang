//TEST:SIMPLE(filecheck=SPIRV): -target spirv
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry vsMain -stage vertex -emit-spirv-via-glsl
//TEST:SIMPLE(filecheck=GLSL): -target glsl -entry vsMain -stage vertex

// SPIRV: OpDecorate %{{.*}} BuiltIn DrawIndex
// GLSL: gl_DrawID

struct VertexIn
{
    float3 position;
    int drawIndex : SV_DrawIndex;
}

[shader("vertex")]
float4 vsMain(VertexIn vin) : SV_Position
{
    return float4(vin.position, vin.drawIndex);
}
