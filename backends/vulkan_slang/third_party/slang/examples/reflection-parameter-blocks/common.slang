// shader.slang

// This module is part of the `reflection-parameter-blocks`
// example program.
//
// This module is split out from the files that define
// individual programs, so that we can share some type
// definitions and utility functions between all of
// the programs and keep them focused on just defining
// the shader entry points.

struct Mesh
{
    float4x4 modelToWorld;
    float4x4 modelToWorld_inverseTranspose;
}

struct Material
{
    Texture2D albedoMap;
    Texture2D glossMap;
    Texture2D normalMap;
    SamplerState sampler;
}

interface ILight
{
}

struct DirectionalLight : ILight
{
    float3 dir;
    float3 intensity;
}

struct ShadowedLight<L : ILight> : ILight
{
    L light;
    Texture2D shadowMap;
    SamplerComparisonState shadowSampler;
    float4x4 worldToShadow;
}

struct EnvironmentMap
{
    TextureCube texture;
    SamplerState sampler;
}

struct Environment
{
    ShadowedLight<DirectionalLight> sunLight;
    EnvironmentMap envMap;
    RWStructuredBuffer<float4> output;
}

struct View
{
    float4x4 worldToView;
    float4x4 viewToProj;
}

// While the Slang compilation library will *reflect* all of
// the shader parameters that a program declares,
// back-ends (such as the SPIR-V code generator) will often
// strip out parameters that are not used as part of the
// computation that a shader performs.
//
// When shader parameters are stripped from the output
// binary code, the runtime system for a particular API
// (e.g., the Vulkan validation layer) cannot check
// whether a program is correctly handling the binding
// of those parameters.
//
// Our example entry points will thus make use of some
// utility routines that serve the purpose of allowing
// us to ensure that specific parameters are seen as
// "used" during code generation.

void use(inout float4 r, float4 v) { r += v; }
void use(inout float4 r, float3 v) { r.xyz += v; }

void use(inout float4 r, Texture2D t, SamplerState s)
{
    use(r, t.SampleLevel(s, r.xy, 0));
}

void use(inout float4 r, RWStructuredBuffer<float4> b)
{
    use(r, b[int(r.x)]);
    b[int(r.x)] = r;
}

void use(inout float4 r, Environment e)
{
    use(r, e.sunLight.light.dir);
    use(r, e.output);
}

void use(inout float4 r, View v)
{
    use(r, v.worldToView[0]);
}

void use(inout float4 r, Material m)
{
    use(r, m.normalMap, m.sampler);
}

void use(inout float4 r, Mesh m)
{
    use(r, m.modelToWorld[0]);
}
