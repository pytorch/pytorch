#version 450
#extension GL_EXT_mesh_shader : require
layout(row_major) uniform;
layout(row_major) buffer;
out gl_MeshPerVertexEXT
{
    vec4 gl_Position;
} gl_MeshVerticesEXT[3];

layout(local_size_x = 3, local_size_y = 1, local_size_z = 1) in;
layout(max_vertices = 3) out;
layout(max_primitives = 1) out;
layout(triangles) out;
void main()
{
    SetMeshOutputsEXT(3U, 1U);
    if(gl_LocalInvocationIndex < 3U)
    {
        gl_MeshVerticesEXT[gl_LocalInvocationIndex].gl_Position.wz = vec2(1.00000000000000000000, 0.00000000000000000000);
        gl_MeshVerticesEXT[gl_LocalInvocationIndex].gl_Position.x = 0.00000000000000000000;
        gl_MeshVerticesEXT[gl_LocalInvocationIndex].gl_Position.y = 0.00000000000000000000;
    }
    else
    {
    }
    if(gl_LocalInvocationIndex < 1U)
    {
        gl_PrimitiveTriangleIndicesEXT[gl_LocalInvocationIndex] = uvec3(0U, 1U, 2U);
    }
    else
    {
    }
    return;
}

