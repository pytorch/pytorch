//TEST:CROSS_COMPILE: -profile sm_6_3 -stage miss -entry main -target spirv-assembly

Texture2DArray  t2D;
SamplerState samplerState;

struct ShadowRay
{
    float hitDistance;
};

void main(in out ShadowRay ray)
{
    float3 uv = float3(ray.hitDistance * 0.2, ray.hitDistance * 0.3f, 0.2f);
    
    float4 val = t2D.SampleGrad(samplerState, uv, float2(0,0), float2(0,0), 0, 0.5f);   
    
    ray.hitDistance = dot(val, val);
}
