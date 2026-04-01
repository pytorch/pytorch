#version 450
// geometry-shader.slang.glsl
//TEST_IGNORE_FILE:

#define RasterVertex RasterVertex_0
#define CoarseVertex CoarseVertex_0

#define input_position  coarseVertices_position_0
#define input_color     coarseVertices_color_0
#define input_id        coarseVertices_id_0
#define output_position outputStream_position_0
#define output_color    outputStream_color_0

layout(row_major) uniform;
layout(row_major) buffer;

layout(location = 0)
in vec4  input_position[3];

layout(location = 1)
in vec3  input_color[3];

layout(location = 2)
in uint  input_id[3];

layout(location = 0)
out vec4 output_position;

layout(location = 1)
out vec3 output_color;

struct RasterVertex
{
    vec4 position_0;
    vec3 color_0;
    uint id_0;
};

struct CoarseVertex
{
    vec4 position_1;
    vec3 color_1;
    uint id_1;
};


layout(max_vertices = 3) out;
layout(triangles) in;
layout(triangle_strip) out;

void main()
{
    uint _S6 = uint(gl_PrimitiveIDIn);

    // TODO: Having to make this copy to transpose things is unfortunate.
    //
    // The front-end should be able to generate code using aggregate
    // types for the input, and/or eliminate the redundant temporary
    // by indexing directly into the sub-arrays.
    //
    CoarseVertex_0 _S7 = { _S1[0], _S2[0], _S3[0] };
    CoarseVertex_0 _S8 = { _S1[1], _S2[1], _S3[1] };
    CoarseVertex_0 _S9 = { _S1[2], _S2[2], _S3[2] };

    CoarseVertex_0  _S10[3] = { _S7, _S8, _S9 };

    int ii_0 = 0;

    for(;;)
    {
        RasterVertex_0 rasterVertex_0;
        rasterVertex_0.position_0 = _S10[ii_0].position_1;
        rasterVertex_0.color_0 = _S10[ii_0].color_1;
        rasterVertex_0.id_0 = _S10[ii_0].id_1 + _S6;
        RasterVertex_0 _S11 = rasterVertex_0;
        _S4 = rasterVertex_0.position_0;
        _S5 = _S11.color_0;
        gl_Layer = int(_S11.id_0);
        EmitVertex();
        int ii_1 = ii_0 + 1;
        if(ii_1 < 3)
        {
        }
        else
        {
            break;
        }
        ii_0 = ii_1;
    }
    return;
}
