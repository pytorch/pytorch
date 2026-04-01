// closesthit.slang

//TEST:SIMPLE(filecheck=GL_SPIRV): -stage anyhit -entry main -target spirv-assembly -emit-spirv-via-glsl
//TEST:SIMPLE(filecheck=SPIRV): -stage anyhit -entry main -target spirv

struct SphereHitAttributes
{
    float3 normal;
};


struct ShadowRay
{
    float4 hitDistance;
    float3 dummyOut;
};

struct Params
{
    Texture2D<float>    alphaMap;
    SamplerState        sampler;
    int                 mode;
}
ParameterBlock<Params> gParams;

void main(
    SphereHitAttributes attributes,
    in out ShadowRay    ioPayload)
{
    if(gParams.mode != 0)
    {
        float val = gParams.alphaMap.SampleLevel(
            gParams.sampler,
            attributes.normal.xy, 0);
        if(val > 0)
        {
            AcceptHitAndEndSearch();
        }
        else
        {
            IgnoreHit();
        }
    }

    uint index = 0U;
    ioPayload.dummyOut = HitTriangleVertexPosition(index);

    index = 1U;
    ioPayload.dummyOut += HitTriangleVertexPosition(index);

    index = 2U;
    ioPayload.dummyOut += HitTriangleVertexPosition(index);
}

// SPIRV-DAG: OpCapability RayTracing
// SPIRV-DAG: OpCapability RayTracingPositionFetchKHR
// SPIRV: OpEntryPoint
// SPIRV: BuiltIn HitTriangleVertexPositionsKHR
// SPIRV: OpTypePointer HitAttribute{{NV|KHR}}
// SPIRV: OpVariable{{.*}}HitAttribute{{NV|KHR}}
// SPIRV: OpIgnoreIntersectionKHR
// SPIRV: OpTerminateRayKHR
// SPIRV-DAG: %{{.*}} = OpAccessChain %{{.*}} %{{.*}} %{{.*}}
// SPIRV-DAG: %{{.*}} = OpAccessChain %{{.*}} %{{.*}} %{{.*}}
// SPIRV-DAG: %{{.*}} = OpAccessChain %{{.*}} %{{.*}} %{{.*}}

// GL_SPIRV-DAG: OpCapability RayTracing
// GL_SPIRV-DAG: OpCapability RayTracingPositionFetchKHR
// GL_SPIRV: OpEntryPoint
// GL_SPIRV: BuiltIn HitTriangleVertexPositionsKHR
// GL_SPIRV-DAG: OpTypePointer HitAttribute{{NV|KHR}}
// GL_SPIRV: OpTerminateRayKHR
// GL_SPIRV: OpIgnoreIntersectionKHR
// GL_SPIRV-DAG: %{{.*}} = OpAccessChain %{{.*}} %{{.*}} %{{.*}}
// GL_SPIRV-DAG: %{{.*}} = OpAccessChain %{{.*}} %{{.*}} %{{.*}}
// GL_SPIRV-DAG: %{{.*}} = OpAccessChain %{{.*}} %{{.*}} %{{.*}}
