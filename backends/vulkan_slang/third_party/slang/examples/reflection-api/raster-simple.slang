// raster-simple.slang

struct AssembledVertex
{
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 uv       : TEXCOORD;
};

struct RasterVertex
{
    float3 worldPosition;
    float3 worldNormal;
    float2 uv;
};

struct Model
{
    float3x4 modelToWorld;
    float3x4 modelToWorld_inverseTranspose;
}

struct Material
{
    Texture2D<float3> albedoMap;
    Texture2D<float3> normalMap;
    Texture2D<float> glossMap;
    SamplerState sampler;
    float2 uvScale;
    float2 uvBias;
}

struct Camera
{
    float3x4 worldToView;
    float3x4 worldToView_inverseTranspose;

    float4x4 viewToProj;
}

struct DirectionalLight
{
    float3 intensity;
    float3 direction;
}

struct Environment
{
    TextureCube environmentMap;
    DirectionalLight light;
}

uniform Model                       model;
uniform ParameterBlock<Material>    material;
uniform ConstantBuffer<Camera>      camera;
uniform ParameterBlock<Environment> environment;

[shader("vertex")]
[require(sm_6_0)]
void vertexMain(
    in AssembledVertex assembledVertex : A,
    out RasterVertex rasterVertex : R,
    in uint vertexID : SV_VertexID,
    out float4 projPosition :  SV_Position)
{
    float3 worldPosition = mul(model.modelToWorld, float4(assembledVertex.position,1));

    rasterVertex.worldPosition = worldPosition;
    rasterVertex.worldNormal = mul(model.modelToWorld_inverseTranspose, float4(assembledVertex.normal,0));
    rasterVertex.uv = assembledVertex.uv;

    float3 viewPosition = mul(camera.worldToView, float4(worldPosition,1));
    projPosition = mul(camera.viewToProj, float4(viewPosition,1));
}

[shader("fragment")]
[require(sm_6_0)]
float4 fragmentMain(
    in RasterVertex vertex : R)
    : SV_Target0
{
    float3 normal = vertex.worldNormal;

    float3 albedo = material.albedoMap.Sample(material.sampler, vertex.uv);

    float3 color = albedo * max(0, dot(normal, environment.light.direction));
    return float4(color, 1);
}
