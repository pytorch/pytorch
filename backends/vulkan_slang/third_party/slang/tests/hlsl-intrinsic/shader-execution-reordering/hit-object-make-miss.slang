// hit-object-make-miss.slang

//TEST:SIMPLE: -target dxil -entry rayGenerationMain -stage raygeneration -profile sm_6_5 -DNV_SHADER_EXTN_SLOT=u0
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry rayGenerationMain -stage raygeneration -profile sm_6_5 -line-directive-mode none 
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry rayGenerationMain -stage raygeneration -emit-spirv-directly

//DISABLE_TEST:COMPARE_COMPUTE_EX:-slang -compute -dx12 -output-using-type -profile sm_6_5 -nvapi-slot u0 
//DISABLE_TEST(compute):COMPARE_COMPUTE:-d3d12 -output-using-type -use-dxil -profile sm_6_5 -render-feature ray-query
//DISABLE_TEST(compute):COMPARE_COMPUTE:-vk -output-using-type -render-feature ray-query

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

void rayGenerationMain()
{
    int idx = DispatchRaysIndex().x;

    RayDesc ray;  
    ray.Origin = float3(idx, 0, 0);
    ray.TMin = 0.01f;
    ray.Direction = float3(0, 1, 0);
    ray.TMax = 1e4f;
    // SPIRV: OpHitObjectRecordMissNV
    HitObject hit = HitObject::MakeMiss(idx, ray);

    // SPIRV: OpHitObjectIsMissNV
    int r = int(hit.IsMiss());
    
    outputBuffer[idx] = r;
}
