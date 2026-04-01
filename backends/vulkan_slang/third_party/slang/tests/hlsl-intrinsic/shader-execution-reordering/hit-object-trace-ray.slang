// hit-object-trace-ray.slang

//TEST:SIMPLE: -target dxil -entry rayGenerationMain -stage raygeneration -profile sm_6_5 -DNV_SHADER_EXTN_SLOT=u0 
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry rayGenerationMain -stage raygeneration -profile sm_6_5 -line-directive-mode none 
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry rayGenerationMain -stage raygeneration -emit-spirv-directly

//DISABLE_TEST(compute):COMPARE_COMPUTE:-d3d12 -output-using-type -use-dxil -profile sm_6_6 -render-feature ray-query
//DISABLE_TEST(compute):COMPARE_COMPUTE:-vk -output-using-type -render-feature ray-query

//TEST_INPUT: set scene = AccelerationStructure
uniform RaytracingAccelerationStructure scene;

//TEST_INPUT:set outputBuffer = out ubuffer(data=[0, 0, 0, 0], stride=4)
RWStructuredBuffer<uint> outputBuffer;

struct SomeValues
{
    int a;
    float b;
};

uint calcValue(HitObject hit)
{
    uint r = 0;
    
    if (hit.IsHit())
    {
        uint instanceIndex = hit.GetInstanceIndex();
        uint instanceID = hit.GetInstanceID();
        uint geometryIndex = hit.GetGeometryIndex();
        uint primitiveIndex = hit.GetPrimitiveIndex();

        SomeValues objSomeValues = hit.GetAttributes<SomeValues>();
        
        r += instanceIndex;
        r += instanceID;
        r += geometryIndex;
        r += primitiveIndex;
        r += objSomeValues.a;
    }
    
    return r;
}

void rayGenerationMain() 
{
    int2 launchID = int2(DispatchRaysIndex().xy);
    int2 launchSize = int2(DispatchRaysDimensions().xy);

    int idx = launchID.x;
    
    SomeValues someValues = { idx, idx * 2.0f };

    RayDesc ray;  
    ray.Origin = float3(idx, 0, 0);
    ray.TMin = 0.01f;
    ray.Direction = float3(0, 1, 0);
    ray.TMax = 1e4f;
    
    RAY_FLAG rayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    uint instanceInclusionMask = 0xff;
    uint rayContributionToHitGroupIndex = 0;
    uint multiplierForGeometryContributionToHitGroupIndex = 4;
    uint missShaderIndex = 0;
    // SPIRV: OpHitObjectTraceRayNV
    HitObject hit = HitObject::TraceRay(scene, 
        rayFlags, 
        instanceInclusionMask, 
        rayContributionToHitGroupIndex, 
        multiplierForGeometryContributionToHitGroupIndex, 
        missShaderIndex, 
        ray, 
        someValues);
    
    outputBuffer[idx] = calcValue(hit);
}
