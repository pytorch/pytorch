//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK: OpEntryPoint

static extern const bool _AlphaTest = false;
static extern const bool _BaseColorMap = false;
static extern const bool _NormalMap = false;
static extern const bool _EmissiveMap = false;
static extern const bool _MetallicRoughnessMap = false;

interface IVertexInput
{
    float3 GetPosition();
    float3 GetNormal();
    float4 GetTangent();
    float2 GetUV();
    float4 GetBone();
}

struct StandardVertexInput : IVertexInput
{
    float3 position;
    float3 normal;
    float4 tangent;
    float2 uv;

    float3 GetPosition() {return position;}
    float3 GetNormal() {return normal;}
    float4 GetTangent() {return tangent;}
    float2 GetUV() { return uv;}
    float4 GetBone() { return float4(0,0,0,0);}
}

extern struct VertexInput : IVertexInput = StandardVertexInput;

struct PushConstant
{
    float4x4 model;
    float4x4 invTspModel;
} 

[vk_push_constant]
PushConstant pconst;

struct VertexOutput
{
    float4 position : SV_Position;
    float3 positionWS;
    float3 normalWS;
    float3 tangentWS;
    float3 bitangentWS;
    float2 uv;
}



VertexOutput SceneLitVertex(VertexInput input)
{
    VertexOutput output;
    float3 vertexPos = input.GetPosition();
    float3 normal = input.GetNormal();
    float4 tangent = input.GetTangent();

    output.positionWS = mul(pconst.model, float4(vertexPos, 1)).xyz;

    matrix<float,3,3> model33 = (float3x3)pconst.invTspModel;
    output.normalWS = mul(model33, normal).xyz;
    output.tangentWS = mul(model33, tangent.xyz);
    output.bitangentWS = tangent.w * cross(output.normalWS, output.tangentWS);
    output.uv = input.GetUV();

    float4x4 vp = {};
    output.position = mul(vp, float4(output.positionWS, 1));

    return output;
}


[shader("vertex")]
VertexOutput VertexMain(VertexInput input)
{
    return SceneLitVertex(input);
}

struct PerMaterial
{
    float4 baseColorFactor;
    float4 emissive;
    float roughness;
    float metallic;
    float alphaCutoff;

    Sampler2D baseColorTex;
    Sampler2D normalMap;
    Sampler2D metallicRoughnessMap;
    Sampler2D emissiveMap;
};

ParameterBlock<PerMaterial> perMaterial;

struct FragmentOutput
{
    float4 color : SV_Target0;
}

FragmentOutput SceneLitFragment(VertexOutput input)
{
    FragmentOutput output;
    output.color = 0;

    return output;
}

[shader("fragment")]
FragmentOutput FragmentMain(VertexOutput input)
{
    return SceneLitFragment(input);
}