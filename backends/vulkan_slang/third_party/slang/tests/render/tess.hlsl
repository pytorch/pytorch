//DISABLED_TEST:COMPARE_HLSL: -profile sm_5_1 -entry HS -stage hull -entry DS -stage domain

// tests/render/tess.hlsl

// -profile vs_5_1 -o my-shader.vs.dxbc -entry VS -profile ps_5_1 -o my-shader.ps.dxbc -entry PS

struct IA_OUTPUT
{
    float3 cpoint : CPOINT;
};

struct VS_OUTPUT
{
    float3 cpoint : CPOINT;
};

struct HS_CONSTANT_OUTPUT
{
    float edges[2] : SV_TessFactor;
};

struct HS_OUTPUT
{
    float3 cpoint : CPOINT;
};

struct DS_OUTPUT
{
    float4 position : SV_Position;
};

VS_OUTPUT VS(IA_OUTPUT input)
{
    VS_OUTPUT output;
    output.cpoint = input.cpoint;
    return output;
}

HS_CONSTANT_OUTPUT HSConst()
{
    HS_CONSTANT_OUTPUT output;

    output.edges[0] = 1.0f; // Detail factor
    output.edges[1] = 64.0f; // Density factor

    return output;
}

[domain("isoline")]
[partitioning("integer")]
[outputtopology("line")]
[outputcontrolpoints(4)]
[maxtessfactor(16.0)]
[patchconstantfunc("HSConst")]
HS_OUTPUT HS(InputPatch<VS_OUTPUT, 4> ip, uint id : SV_OutputControlPointID)
{
    HS_OUTPUT output;
    output.cpoint = ip[id].cpoint;
    return output;
}

[domain("isoline")]
DS_OUTPUT DS(HS_CONSTANT_OUTPUT input, OutputPatch<HS_OUTPUT, 4> op, float2 uv : SV_DomainLocation)
{
    DS_OUTPUT output;

    float t = uv.x;

    float3 pos = pow(1.0f - t, 3.0f) * op[0].cpoint + 3.0f * pow(1.0f - t, 2.0f) * t * op[1].cpoint + 3.0f * (1.0f - t) * pow(t, 2.0f) * op[2].cpoint + pow(t, 3.0f) * op[3].cpoint;

    output.position = float4(pos, 1.0f);

    return output;
}

float4 PS(DS_OUTPUT input) : SV_Target0
{
    return float4(0.0f, 0.0f, 0.0f, 1.0f);
}
