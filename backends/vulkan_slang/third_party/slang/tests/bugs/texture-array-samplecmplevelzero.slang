//TEST_DISABLED:SIMPLE(filecheck=CHECK): -entry MainPs -stage fragment -profile glsl_450 -target spirv
//CHECK: OpEntryPoint

SamplerComparisonState g_tSFMShadowDepthTexture_sampler; 
Texture2DArray g_tSFMShadowDepthTexture; 

struct PS_INPUT
{
  float3 vTexCoord : TEXCOORD0;
};

struct PS_OUTPUT
{
  float4 vColor : SV_Target0;
};

PS_OUTPUT MainPs ( PS_INPUT i ) 
{ 
  PS_OUTPUT o;
  float flDepth = 0.5;
  o.vColor = g_tSFMShadowDepthTexture.SampleCmpLevelZero ( g_tSFMShadowDepthTexture_sampler, i.vTexCoord.xyz, flDepth ).rrrr;
  return o;
} 