//TEST:SIMPLE(filecheck=HULL):   -target spirv -stage hull   -entry hullMain
//TEST:SIMPLE(filecheck=DOMAIN): -target spirv -stage domain -entry domainMain

// Testing if `InputPatch` and `OutputPatch` can be used as function arguments.
// `[ForceInline]` can be used to workaround but it should work without it.

// HULL: OpEntryPoint TessellationControl %hullMain
// HULL: = OpVariable %{{[a-zA-Z_0-9]*}} Input

// DOMAIN: OpEntryPoint TessellationEvaluation %domainMain
// DOMAIN: = OpVariable %{{[a-zA-Z_0-9]*}} Output

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


VS_OUT GetInputPatch(InputPatch<VS_OUT, 4> patch, int index)
{
    return patch[index];
}

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
    float3 p0 = GetInputPatch(patch, 0).position;
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

HS_OUT GetOutputPatch(const OutputPatch<HS_OUT, 4> patch, int index)
{
    return patch[index];
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
    float3 p0 = GetOutputPatch(patch, 0).position;
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
