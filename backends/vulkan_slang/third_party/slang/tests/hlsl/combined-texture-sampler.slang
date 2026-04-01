//TEST:SIMPLE(filecheck=CHECK): -target hlsl -entry fragMain -profile ps_6_0
//TEST:SIMPLE(filecheck=DXIL): -target dxil -entry fragMain -profile ps_6_0

// Check that we can correctly cross compile combined texture samplers to HLSL.

// DXIL: define void @fragMain()

// CHECK-DAG: Texture2D<float4 > sampler_texture{{.*}} : register(t0);
// CHECK-DAG: SamplerState sampler_sampler{{.*}} : register(s0);
Sampler2D sampler;

// CHECK-DAG: Texture2D<float4 > followingTexture{{.*}} : register(t1);
Texture2D followingTexture;

// CHECK-DAG: SamplerState followingSamplerState{{.*}} : register(s1);
SamplerState followingSamplerState;

// CHECK-DAG: Texture2D<float4 > explicitBindingSampler_texture{{.*}} : register(t4);
// CHECK-DAG: SamplerState explicitBindingSampler_sampler{{.*}} : register(s3);
Sampler2D explicitBindingSampler : register(t4): register(s3);

float4 fragMain() : SV_Target
{
    return sampler.Sample(float2(0, 0)) + explicitBindingSampler.Sample(float2(0, 0)) + followingTexture.Sample(followingSamplerState, float2(0,0));
}