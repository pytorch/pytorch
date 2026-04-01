#pragma pack_matrix(column_major)
#ifdef SLANG_HLSL_ENABLE_NVAPI
#include "nvHLSLExtns.h"
#endif
#pragma warning(disable: 3557)

static const float2  positions_0[int(3)] = { float2(0.0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5) };
static const float3  colors_0[int(3)] = { float3(1.0, 1.0, 0.0), float3(0.0, 1.0, 1.0), float3(1.0, 0.0, 1.0) };
struct Vertex_0
{
    float4 pos_0 : SV_Position;
    float3 color_0 : Color;
};

[shader("mesh")][numthreads(3, 1, 1)]
[outputtopology("triangle")]
void main(uint tig_0 : SV_GROUPINDEX, vertices vertices out Vertex_0  verts_0[int(3)], indices indices out uint3  triangles_0[int(1)])
{
    SetMeshOutputCounts(3U, 1U);
    if(tig_0 < 3U)
    {
        verts_0[tig_0].pos_0 = float4(positions_0[tig_0], 0.0, 1.0);
        verts_0[tig_0].color_0 = colors_0[tig_0];
    }
    else
    {
    }

    if(tig_0 < 1U)
    {
        triangles_0[tig_0] = uint3(0U, 1U, 2U);
    }
    else
    {
    }

    return;
}
