//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):-target spirv

struct Light
{
    float3 position;
    float radius;

    float3 color;
    float intensity;
};

[vk::binding(0, 0)]
StructuredBuffer<Light> globalLightList;

struct Lighting
{
    //CHECK: ([[# @LINE+1]]): error 20102
    float3 DoLighting(Light light);
    {
        // Not emitted
        return float3(1.0, 1.0, 1.0);
    }
};

[shader("fragment")]
float4 fragment(float4 color: COLOR0)
{
    float4 albedo = color;

    if (albedo.a < 0.025)
        discard;

    Lighting light = Lighting();
    albedo.xyz = light.DoLighting(globalLightList[0]);

    return albedo;
}