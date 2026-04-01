//TEST_CATEGORY(wave, compute)
//DISABLE_TEST:COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//DISABLE_TEST:COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST:COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_0 -shaderobj -render-feature hardware-device
//DISABLE_TEST(vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST:COMPARE_COMPUTE_EX:-cuda -compute -render-features cuda_sm_7_0 -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(8, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int idx = int(dispatchThreadID.x);
    
    // NOTE! dxc only supports bit ops on uint and associated types NOT int
    // Also GLSL does not have built in support for int matrices. So we'll just try with float for now    
    // GLSL does not support matrix types for Wave like intrinsics   
       
    matrix<int, 2, 2> v0 = matrix<int, 2, 2>(idx + 1, idx + 2, idx + 3, idx + 4);   
    matrix<float, 2, 2> v1 = matrix<float, 2, 2>(v0) + matrix<float, 2, 2>(1, 1, 1, 1);
    
    
    matrix<uint, 2, 2> uv0 = matrix<uint, 2, 2>(v0[0][0], v0[0][1], v0[1][0], v0[0][1]);
    
    matrix<int, 2, 2> r0 = WaveActiveSum(v0);
    matrix<float, 2, 2> r1 = WaveActiveSum(v1);
    matrix<uint, 2, 2> r2 = WaveActiveBitXor(uv0);
    matrix<uint, 2, 2> r3 = WaveActiveBitOr(uv0);
    matrix<uint, 2, 2> r4 = WaveActiveBitAnd(uv0);
    
    matrix<uint, 2, 2> r5 = r2 + r3 + r4;
    matrix<int, 2, 2> r6 = matrix<int, 2, 2>(int(r5[0][0]), int(r5[0][1]), int(r5[1][0]), int(r5[1][1])); 
    
    matrix<int, 2, 2> r = r0 + matrix<int, 2, 2>(r1) + r6;
   
    outputBuffer[idx] = r[0][0] + r[0][1] + r[1][0] + r[1][1];
}