// hit-object-output.slang

// This test validates that `HitObject`s can be used
// as function results (including `out` parameters)

//TEST:SIMPLE: -target dxil -entry rayGenerationMain -stage raygeneration -profile sm_6_5 -DNV_SHADER_EXTN_SLOT=u0
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry rayGenerationMain -stage raygeneration -profile sm_6_5 -line-directive-mode none 
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry rayGenerationMain -stage raygeneration -emit-spirv-directly

//DISABLE_TEST(compute):COMPARE_COMPUTE:-d3d12 -output-using-type -use-dxil -profile sm_6_5 -render-feature ray-query
//DISABLE_TEST(compute):COMPARE_COMPUTE:-vk -output-using-type -render-feature ray-query

//TEST_INPUT: set scene = AccelerationStructure
uniform RaytracingAccelerationStructure scene;

//TEST_INPUT:set outputBuffer = out ubuffer(data=[0, 0, 0, 0], stride=4)
RWStructuredBuffer<uint> outputBuffer;

struct MyAttributes
{
    uint value;
};

struct MyRayPayload
{
    uint value;
};

void accumulate(inout uint value, HitObject hit)
{
    value = value*256;

    if (hit.IsHit())
    {
        value += 16 + hit.GetAttributes<MyAttributes>().value;
    }
}

RayDesc makeRay(uint idx, uint variation)
{
    RayDesc ray;  
    ray.Origin = float3(idx, 0, variation);
    ray.TMin = 0.01f;
    ray.Direction = float3(0, 1, 0);
    ray.TMax = 1e4f;
    return ray;
}

HitObject myTraceRay(uint idx)
{
    MyRayPayload payload = { idx };

    RayDesc ray = makeRay(idx, 0);
    
    RAY_FLAG rayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    uint instanceInclusionMask = 0xff;
    uint rayContributionToHitGroupIndex = 0;
    uint multiplierForGeometryContributionToHitGroupIndex = 4;
    uint missShaderIndex = 0;

    // SPIRV-DAG: OpHitObjectTraceRayNV
    return HitObject::TraceRay(scene,
        rayFlags, 
        instanceInclusionMask, 
        rayContributionToHitGroupIndex, 
        multiplierForGeometryContributionToHitGroupIndex, 
        missShaderIndex, 
        ray, 
        payload);
}

void copyHitObjectHandle(
    out HitObject dst,
    HitObject src)
{
    dst = src;
}

void myMakeMiss(
    uint idx,
    inout HitObject h)
{
    RayDesc ray = makeRay(idx, 1);
    h = HitObject::MakeMiss(idx, ray);
}

void rayGenerationMain() 
{
    uint idx = uint(DispatchRaysIndex().x);

    uint r = 0;
    
    HitObject hit = myTraceRay(idx);

    accumulate(r, hit);

#if 0 // cannot support this right now
    HitObject hit2;
    copyHitObjectHandle(hit2, hit);

    accumulate(r, hit2);
#else
    accumulate(r, hit);

#endif

    myMakeMiss(idx, hit);

    accumulate(r, hit);
    accumulate(r, hit);

    outputBuffer[idx] = r;
}
