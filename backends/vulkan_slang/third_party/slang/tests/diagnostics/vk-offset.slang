//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -entry vertexMain -stage vertex

// CHECK: error 31002: attribute 'vk_offset' is not valid here
// CHECK-NEXT: [vk::offset(16)]] struct S1
// CHECK: error 31002: attribute 'vk_offset' is not valid here
// CHECK-NEXT: [vk::offset(8)]] VertexOutput output;

[[vk::offset(16)]] struct S1
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

[shader("vertex")]
VertexOutput vertexMain()
{
    [[vk::offset(8)]] VertexOutput output;
    output.position = float3(pc.a.x, pc.b.y, pc.c.z);
    return output;
}

