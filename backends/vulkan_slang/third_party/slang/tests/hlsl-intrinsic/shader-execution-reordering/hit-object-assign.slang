// hit-object-assign.slang

//TEST:SIMPLE: -target dxil -entry rayGenerationMain -stage raygeneration -profile sm_6_5 -DNV_SHADER_EXTN_SLOT=u0 
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry rayGenerationMain -stage raygeneration -profile glsl_460+GL_EXT_ray_tracing -O0 -line-directive-mode none 
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry rayGenerationMain -stage raygeneration -O0  -emit-spirv-directly

//DISABLE_TEST:COMPARE_COMPUTE_EX:-slang -compute -dx12 -output-using-type -profile sm_6_5 -nvapi-slot u0 
//DISABLE_TEST(compute):COMPARE_COMPUTE:-d3d12 -output-using-type -use-dxil -profile sm_6_5 -render-feature ray-query
//DISABLE_TEST(compute):COMPARE_COMPUTE:-vk -output-using-type -render-feature ray-query

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

// SPIRV: OpHitObjectRecordMissNV
// SPIRV: OpHitObjectIsMissNV

void rayGenerationMain()
{    
    int2 launchID = int2(DispatchRaysIndex().xy);
    int2 launchSize = int2(DispatchRaysDimensions().xy);

    int idx = int(launchID.x);

    RayDesc ray;  
    ray.Origin = float3(idx, 0, 0);
    ray.TMin = 0.01f;
    ray.Direction = float3(0, 1, 0);
    ray.TMax = 1e4f;
    
    HitObject hit = HitObject::MakeMiss(idx, ray);
    
    // Let's try assigning. This should work on VK
    // because it will SSA out such there isn't an assignment in GLSL output.
    hit = HitObject::MakeMiss(idx + 1, ray);
    
    int r = int(hit.IsMiss());
    
    outputBuffer[idx] = r;
}
