//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage domain

// CHECK: TessLevelInner

struct HS_CONTROL_POINT_OUTPUT
{
    float4 Position : POSITION;
    float2 TexCoord : TEXCOORD;
};

struct HS_CONSTANT_DATA_OUTPUT
{
    float EdgeTessFactor[4] : SV_TessFactor;
    float InsideTessFactor[2] : SV_InsideTessFactor;
};


struct DS_OUTPUT
{
    float4 Position : SV_Position;
    float2 TexCoord : TEXCOORD;
};

// Domain Shader (DS)
[domain("quad")]
DS_OUTPUT main(HS_CONSTANT_DATA_OUTPUT input, const OutputPatch<HS_CONTROL_POINT_OUTPUT, 4> patch, float2 uv : SV_DomainLocation)
{
    DS_OUTPUT output;
    output.Position = input.EdgeTessFactor[0];
    output.TexCoord = input.InsideTessFactor[0];
    return output;
}