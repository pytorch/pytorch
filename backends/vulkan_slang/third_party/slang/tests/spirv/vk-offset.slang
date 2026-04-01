//TEST:SIMPLE(filecheck=CHECK_SPV): -target spirv -emit-spirv-directly -entry vertexMain -stage vertex
//TEST:SIMPLE(filecheck=CHECK_SPV): -target spirv -emit-spirv-via-glsl -entry vertexMain -stage vertex
//TEST:SIMPLE(filecheck=CHECK_GLSL): -target glsl -entry vertexMain -stage vertex

struct S1
{
    [[vk::offset(32)]]
    float2 a;

    float3 b;

    [[vk::offset(16)]]
    float4 c;
  }

[[vk::push_constant]]
S1 pc;

struct VertexOutput
{
    float3 position : SV_Position;
}

// CHECK_SPV: OpDecorate [[STRUCT:%[a-zA-Z0-9_]+]] Block
// CHECK_SPV-NEXT: OpMemberDecorate [[STRUCT]] 0 Offset 32
// CHECK_SPV-NEXT: OpMemberDecorate [[STRUCT]] 1 Offset 48
// CHECK_SPV-NEXT: OpMemberDecorate [[STRUCT]] 2 Offset 16

// CHECK_GLSL: layout(std430) uniform
// CHECK_GLSL-NEXT: {
// CHECK_GLSL-NEXT: layout(offset = 32) vec2 a
// CHECK_GLSL-NEXT: vec3 b
// CHECK_GLSL-NEXT: layout(offset = 16) vec4 c

[shader("vertex")]
VertexOutput vertexMain()
{
    VertexOutput output;
    output.position = float3(pc.a.x, pc.b.y, pc.c.z);
    return output;
}

