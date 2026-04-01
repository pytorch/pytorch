// get-attribute-at-vertex.slang.glsl
//TEST_IGNORE_FILE:

#version 450
#extension GL_EXT_fragment_shader_barycentric : require
layout(row_major) uniform;
layout(row_major) buffer;

pervertexEXT layout(location = 0)
in vec4  color_0[3];

layout(location = 0)
out vec4 result_0;

void main()
{
    result_0 = gl_BaryCoordEXT.x * ((color_0)[(0U)]) + gl_BaryCoordEXT.y * ((color_0)[(1U)]) + gl_BaryCoordEXT.z * ((color_0)[(2U)]);
    return;
}
