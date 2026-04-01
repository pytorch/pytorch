// image-load.slang.glsl
//TEST_IGNORE_FILE:

#version 450

#extension GL_EXT_samplerless_texture_functions : require

layout(r32f)
layout(binding = 0)
uniform image2DArray gParams_tex_0;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
    float _S1 = imageLoad(
    	gParams_tex_0,
    	ivec3(
    		ivec2(gl_GlobalInvocationID.xy),
    		int(gl_GlobalInvocationID.z))).x;

    return;
}
