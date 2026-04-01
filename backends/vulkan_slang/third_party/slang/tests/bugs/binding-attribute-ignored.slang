// binding-attribute-ignored.slang
// Test that binding attributes on uniforms that get packed into the default uniform buffer trigger a warning

//TEST:SIMPLE(filecheck=CHECK):-target spirv

//CHECK: ([[# @LINE+2]]): warning 39071
[[vk::binding(1, 2)]]
uniform float4 g_position;

//CHECK: ([[# @LINE+2]]): warning 39071
[[vk::binding(3, 1)]]
uniform float4x4 g_transform;

// This won't trigger a warning because it's a texture (not packed into default uniform buffer)
[[vk::binding(0, 0)]]
Texture2D g_texture;

[shader("vertex")]
float4 main(float4 pos : POSITION) : SV_POSITION
{
    return g_position;
} 