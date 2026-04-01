//TEST:SIMPLE(filecheck=CHECK_SPV): -emit-spirv-directly -stage raygeneration -entry main -target spirv-assembly
//TEST:SIMPLE(filecheck=CHECK_HLSL): -stage raygeneration -entry main -target hlsl
//TEST:SIMPLE(filecheck=CHECK_GLSL): -stage raygeneration -entry main -target glsl

#define TRACING_EPSILON 1e-6

Texture2D   samplerPosition;
Texture2D   samplerNormal;
SamplerState sampler;

struct Light {
    float4 position;
    float4 color;
};

struct Uniforms
{
    Light       light;
    float4      viewPos;
    float4x4    view;
    float4x4    model;    
};
ConstantBuffer<Uniforms> ubo;

layout(rgba32f)
RWTexture2D<float4> outputImage;

RaytracingAccelerationStructure as;

struct ShadowRay
{
    float hitDistance;
};

struct ReflectionRay
{
    float color;
};

void main() 
{
    int2 launchID = int2(DispatchRaysIndex().xy);
    int2 launchSize = int2(DispatchRaysDimensions().xy);
    
    float2 inUV = float2(
        (float(launchID.x) + 0.5f) / float(launchSize.x),
        (float(launchID.y) + 0.5f) / float(launchSize.y)
    );

    float3 P = samplerPosition.SampleLevel(sampler, inUV, 0).rgb;
    float3 N = samplerNormal.SampleLevel(sampler, inUV, 0).rgb * 2.0 - 1.0;

    float3 lightPos = ubo.light.position.xyz;
    float3 lightDelta = lightPos - P;
    float lightDist = length(lightDelta);
    float3 L = normalize(lightDelta);
    float atten = 1.0f / (lightDist*lightDist);

    RayDesc ray;
    ray.Origin = P;
    ray.TMin = TRACING_EPSILON;
    ray.Direction = lightDelta;
    ray.TMax = lightDist;

    {
        ShadowRay shadowRay;
        shadowRay.hitDistance = 0;

        float currentTime = 1;

        TraceMotionRay(as,
                // ray flags
                1,
                // cull mask
                0xff,
                // sbt record offset
                0,
                // sbt record stride
                0,
                // missIndex
                2,
                // ray
                ray,
                // currentTime
                currentTime,
                // payload
                shadowRay);
        
        if (shadowRay.hitDistance < lightDist)
        {
            atten = 0.f;
        }
    }

    float3 color = ubo.light.color.xyz * saturate(dot(N,L)) * atten;

    {
        ReflectionRay reflectionRay;
        TraceRay(as,
                // ray flags
                1,
                // cull mask
                0xff,
                // sbt record offset
                0,
                // sbt record stride
                0,
                // missIndex
                2,
                // ray
                ray,
                // payload
                reflectionRay);


        color = color + reflectionRay.color;
    }

    outputImage[launchID] = float4(color, 1.0);
}

// CHECK_SPV: %{{.*}} = OpVariable %_ptr_RayPayload{{NV|KHR}}_ReflectionRay{{.*}} RayPayload
// CHECK_SPV: OpTraceRayMotionNV
// CHECK_SPV: OpTraceRayKHR

// CHECK_HLSL: TraceMotionRay
// CHECK_HLSL: TraceRay

// CHECK_GLSL: traceRayMotionNV(
// CHECK_GLSL: traceRayEXT(
