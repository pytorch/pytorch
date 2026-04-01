//TEST:SIMPLE(filecheck=CHECK_SPV):-target spirv -entry computeMain -stage compute -emit-spirv-directly
//TEST:SIMPLE(filecheck=CHECK_SPV):-target spirv -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CHECK_WGSL):-target wgsl -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CHECK_METAL):-target metal -entry computeMain -stage compute

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(8, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);
    
    half2 v1 = half2(1.0h, half(1 << idx));
    
    // CHECK_SPV: OpGroupNonUniformFAdd
    // CHECK_WGSL: subgroupExclusiveAdd
    // CHECK_METAL: simd_prefix_exclusive_sum
    float2 r1 = WavePrefixSum(v1);
    
    outputBuffer[idx] = (int)r1.x;
}
