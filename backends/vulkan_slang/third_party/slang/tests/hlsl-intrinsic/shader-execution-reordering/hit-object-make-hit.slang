// hit-object-make-hit.slang

//TEST:SIMPLE: -target dxil -entry rayGenerationMain -stage raygeneration -profile sm_6_5 -DNV_SHADER_EXTN_SLOT=u0 
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry rayGenerationMain -stage raygeneration -profile sm_6_5 -line-directive-mode none -O0
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry rayGenerationMain -stage raygeneration -emit-spirv-directly 
//DISABLE_TEST(compute):COMPARE_COMPUTE:-d3d12 -output-using-type -use-dxil -profile sm_6_5 -render-feature ray-query
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
    // SPIRV-DAG: OpHitObjectIsHitNV
    // SPIRV-DAG: OpHitObjectGetInstanceCustomIndexNV
    // SPIRV-DAG: OpHitObjectGetInstanceIdNV
    // SPIRV-DAG: OpHitObjectGetGeometryIndexNV
    // SPIRV-DAG: OpHitObjectGetPrimitiveIndexNV
    // SPIRV-DAG: OpHitObjectGetHitKindNV
    // SPIRV-DAG: OpHitObjectIsMissNV
    uint r = 0;
    
    if (hit.IsHit())
    {
        uint instanceIndex = hit.GetInstanceIndex();
        uint instanceID = hit.GetInstanceID();
        uint geometryIndex = hit.GetGeometryIndex();
        uint primitiveIndex = hit.GetPrimitiveIndex();
        uint hitKind = hit.GetHitKind();
        
        r += hitKind;
        r += instanceIndex;
        r += instanceID;
        r += geometryIndex;
        r += primitiveIndex;

        RayDesc ray = hit.GetRayDesc();

        r += uint(ray.TMin > 0);
        r += uint(ray.TMax < ray.TMin);
        
        SomeValues objSomeValues = hit.GetAttributes<SomeValues>();
        
        r += objSomeValues.a;
    }
    else if (hit.IsMiss())
    {
        r += 1;
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
    
    uint hitKind = 0;

    uint r = 0;
    {
        // SPIRV-DAG: OpHitObjectRecordHitNV
        HitObject hit = HitObject::MakeHit(0, scene, idx, idx * 2, idx * 3, hitKind, ray, someValues);
        
        r = calcValue(hit);
    }
    
    {
        int rayContributionToHitGroupIndex = 0;
        int multiplierForGeometryContributionToHitGroupIndex = 4;
        
        HitObject hit = HitObject::MakeHit(scene, 
            idx, 
            idx * 2, 
            idx * 3, 
            hitKind, 
            rayContributionToHitGroupIndex, 
            multiplierForGeometryContributionToHitGroupIndex, 
            ray, 
            someValues);
        
        r += calcValue(hit);
    }
    
    outputBuffer[idx] = r;
}
