#version 450
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_nonuniform_qualifier : require
layout(row_major) uniform;
layout(row_major) buffer;
layout(binding = 0)
uniform texture2D  gParams_textures_0[10];

float fetchData_0(uvec2 coords_0, uint index_0)
{
    return (texelFetch((gParams_textures_0[nonuniformEXT(index_0)]), ivec2((coords_0)), 0).x);
}

layout(location = 0)
out vec4 _S1;

flat layout(location = 0)
in uvec3 _S2;

void main()
{
    _S1 = vec4(fetchData_0(_S2.xy, _S2.z));
    return;
}

