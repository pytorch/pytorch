//TEST:SIMPLE(filecheck=HULL):   -target spirv -stage hull   -entry hullMain
//TEST:SIMPLE(filecheck=DOMAIN): -target spirv -stage domain -entry domainMain

// HULL-DAG: OpExecutionMode %hullMain SpacingEqual
// HULL-DAG: OpExecutionMode %hullMain OutputVertices 4
// HULL-DAG: OpExecutionMode %hullMain VertexOrderCw
// HULL-DAG: OpExecutionMode %hullMain Quads

// HULL: OpDecorate %gl_TessLevelOuter BuiltIn TessLevelOuter
// HULL: OpDecorate %gl_TessLevelOuter Patch
// HULL: OpDecorate %gl_TessLevelInner BuiltIn TessLevelInner
// HULL: OpDecorate %gl_TessLevelInner Patch

// HULL: OpControlBarrier %uint_2 %uint_4 %uint_0

// HULL: OpStore %gl_TessLevelOuter
// HULL: OpStore %gl_TessLevelInner

// DOMAIN-DAG: OpExecutionMode %domainMain SpacingEqual
// DOMAIN-DAG: OpExecutionMode %domainMain Quads

struct VS_OUT
{
    float3 position : POSITION;
};

struct HS_OUT
{
    float3 position : POSITION;
};

struct HSC_OUT
{
    float EdgeTessFactor[4] : SV_TessFactor;
    float InsideTessFactor[2] : SV_InsideTessFactor;
};

struct DS_OUT
{
    float4 position : SV_Position;
};

// Hull Shader (HS)
[domain("quad")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(4)]
[patchconstantfunc("constants")]
HS_OUT hullMain(InputPatch<VS_OUT, 4> patch, uint i : SV_OutputControlPointID)
{
    HS_OUT o;
    o.position = patch[i].position;
    return o;
}

HSC_OUT constants(InputPatch<VS_OUT, 4> patch)
{
    float3 p0 = patch[0].position;
    float3 p1 = patch[1].position;
    float3 p2 = patch[2].position;
    float3 p3 = patch[3].position;

    HSC_OUT o;
    o.EdgeTessFactor[0] = dot(p0, p1); 
    o.EdgeTessFactor[1] = dot(p0, p3);
    o.EdgeTessFactor[2] = dot(p2, p3);
    o.EdgeTessFactor[3] = dot(p1, p2);
    o.InsideTessFactor[0] = lerp(o.EdgeTessFactor[1], o.EdgeTessFactor[3], 0.5);
    o.InsideTessFactor[1] = lerp(o.EdgeTessFactor[0], o.EdgeTessFactor[2], 0.5);
    return o;
}

[domain("quad")]
DS_OUT domainMain(
    float2 uv : SV_DomainLocation,      // Tessellated coordinates (u, v)
    const OutputPatch<HS_OUT, 4> patch, // Control points from the hull shader
    const HSC_OUT patchConstants        // Patch constants calculated by the hull shader
)
{
    DS_OUT o;

    // Interpolate the position of the tessellated point within the patch
    float3 p0 = patch[0].position;
    float3 p1 = patch[1].position;
    float3 p2 = patch[2].position;
    float3 p3 = patch[3].position;

    // Bilinear interpolation of the position in the quad
    float3 interpolatedPosition = 
          p0 * (1 - uv.x) * (1 - uv.y)
        + p1 * uv.x * (1 - uv.y)
        + p3 * uv.x * uv.y
        + p2 * (1 - uv.x) * uv.y;

    // Output final position in clip space
    o.position = float4(interpolatedPosition, 1.0);
    return o;
}
