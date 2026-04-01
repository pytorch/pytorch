#version 450
#extension GL_EXT_mesh_shader : require
layout(row_major) uniform;
layout(row_major) buffer;
struct Texes_0
{
    vec2 tex1_0;
    vec4 tex2_0;
};

struct Vertex_0
{
    vec4 pos_0;
    vec3 col_0;
    Texes_0 ts_0;
};

void just_two_0(out Vertex_0 v_0, out Vertex_0 w_0)
{
    const vec4 _S1 = vec4(0.0);
    const vec3 _S2 = vec3(1.0);
    Texes_0 _S3 = { vec2(0.0, 0.0), vec4(0.0, 0.0, 0.0, 0.0) };
    v_0.pos_0 = _S1;
    v_0.col_0 = _S2;
    v_0.ts_0 = _S3;
    w_0.pos_0 = _S1;
    w_0.col_0 = _S2;
    w_0.ts_0 = _S3;
    return;
}

void just_one_0(out Vertex_0 v_1)
{
    const vec3 _S4 = vec3(1.0);
    Texes_0 _S5 = { vec2(0.0, 0.0), vec4(0.0, 0.0, 0.0, 0.0) };
    v_1.pos_0 = vec4(0.0);
    v_1.col_0 = _S4;
    v_1.ts_0 = _S5;
    return;
}

void part_of_one_0(out vec4 p_0)
{
    p_0 = vec4(1.0, 2.0, 3.0, 4.0);
    return;
}

void write_struct_0(out Texes_0 t_0)
{
    t_0.tex1_0 = vec2(0.0);
    t_0.tex2_0 = vec4(1.0);
    return;
}

layout(location = 0)
out vec3  _S6[3];

layout(location = 1)
out vec2  _S7[3];

layout(location = 2)
out vec4  _S8[3];

out gl_MeshPerVertexEXT
{
    vec4 gl_Position;
} gl_MeshVerticesEXT[3];

out uvec3  gl_PrimitiveTriangleIndicesEXT[1];
void everything_0()
{
    vec3 _S9 = vec3(1.0);
    vec2 _S10 = vec2(0.0, 0.0);
    vec4 _S11 = vec4(0.0, 0.0, 0.0, 0.0);
    gl_MeshVerticesEXT[0U].gl_Position = vec4(0.0);
    _S6[0U] = _S9;
    _S7[0U] = _S10;
    _S8[0U] = _S11;
    return;
}

void a_0()
{
    everything_0();
    return;
}

void b_0()
{
    Vertex_0 _S12;
    Vertex_0 _S13;
    just_two_0(_S13, _S12);
    Vertex_0 _S14 = _S13;
    gl_MeshVerticesEXT[0U].gl_Position = _S13.pos_0;
    _S6[0U] = _S14.col_0;
    _S7[0U] = _S14.ts_0.tex1_0;
    _S8[0U] = _S14.ts_0.tex2_0;
    Vertex_0 _S15 = _S12;
    gl_MeshVerticesEXT[0U].gl_Position = _S12.pos_0;
    _S6[0U] = _S15.col_0;
    _S7[0U] = _S15.ts_0.tex1_0;
    _S8[0U] = _S15.ts_0.tex2_0;
    return;
}

void c_0(uint _S16)
{
    Vertex_0 _S17;
    just_one_0(_S17);
    Vertex_0 _S18 = _S17;
    gl_MeshVerticesEXT[_S16].gl_Position = _S17.pos_0;
    _S6[_S16] = _S18.col_0;
    _S7[_S16] = _S18.ts_0.tex1_0;
    _S8[_S16] = _S18.ts_0.tex2_0;
    return;
}

void d_0(uint _S19)
{
    Vertex_0 _S20;
    Vertex_0 _S21;
    just_two_0(_S21, _S20);
    Vertex_0 _S22 = _S21;
    gl_MeshVerticesEXT[_S19].gl_Position = _S21.pos_0;
    _S6[_S19] = _S22.col_0;
    _S7[_S19] = _S22.ts_0.tex1_0;
    _S8[_S19] = _S22.ts_0.tex2_0;
    Vertex_0 _S23 = _S20;
    gl_MeshVerticesEXT[0U].gl_Position = _S20.pos_0;
    _S6[0U] = _S23.col_0;
    _S7[0U] = _S23.ts_0.tex1_0;
    _S8[0U] = _S23.ts_0.tex2_0;
    return;
}

void e_0(uint _S24)
{
    part_of_one_0(gl_MeshVerticesEXT[_S24].gl_Position);
    Texes_0 _S25;
    write_struct_0(_S25);
    Texes_0 _S26 = _S25;
    _S7[_S24] = _S25.tex1_0;
    _S8[_S24] = _S26.tex2_0;
    part_of_one_0(_S8[_S24]);
    return;
}

layout(local_size_x = 3, local_size_y = 1, local_size_z = 1) in;
layout(max_vertices = 3) out;
layout(max_primitives = 1) out;
layout(triangles) out;
void main()
{
    SetMeshOutputsEXT(3U, 1U);
    if(gl_LocalInvocationIndex < 3U)
    {
        a_0();
        b_0();
        c_0(gl_LocalInvocationIndex);
        d_0(gl_LocalInvocationIndex);
        e_0(gl_LocalInvocationIndex);
    }
    if(gl_LocalInvocationIndex < 1U)
    {
        gl_PrimitiveTriangleIndicesEXT[gl_LocalInvocationIndex] = uvec3(0U, 1U, 2U);
    }
    return;
}

