//TEST:SIMPLE(filecheck=CHECK): -target spirv -entry computeMain -stage compute -emit-spirv-directly

// Check that we are not generating spirv that loads a global resource array into a SSA register,
// instead, these arrays should always be accessed via direct OpAccessChain operations to avoid
// creating a lot of local load/stores in the driver compiler.

struct Scene
{
    SamplerState samplers[256];
    Texture2D textures[100];
}

ParameterBlock<Scene> scene;
struct Material
{
    int sampler;
    int texture;
}

RWStructuredBuffer<float4> result;

float4 shade(Scene scene, Material mat)
{
    return scene.textures[mat.texture].SampleLevel(scene.samplers[mat.sampler], float2(0,0), 0);
}

[numthreads(1,1,1)]
void computeMain(uniform Material mat)
{
    // CHECK: OpEntryPoint
    // CHECK-NOT: OpLoad {{.*}} %scene{{.*}}samplers
    // CHECK-NOT: OpLoad {{.*}} %scene{{.*}}textures
    result[0] = shade(scene, mat);
}