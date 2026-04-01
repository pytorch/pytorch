// Native half not supported on CPU currently
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj
// Doesn't work on DX11 currently - locks up on binding
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -output-using-type -shaderobj
// Produces a different result on DX12 with DXBC than expected(!). So disabled for now
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -output-using-type -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile cs_6_0 -use-dxil -output-using-type -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -output-using-type -shaderobj -render-features half

//TEST_INPUT: RWTexture2D(format=R16Float, size=4, content = one, mipMaps = 1):name rwt2D
[format("r16f")]
RWTexture2D<float> rwt2D;

//TEST_INPUT: RWTexture2D(format=RG16Float, size=4, content = one, mipMaps = 1):name rwt2D_2
[format("rg16f")]
RWTexture2D<float2> rwt2D_2;

//TEST_INPUT: RWTexture2D(format=RGBA16Float, size=4, content = one, mipMaps = 1):name rwt2D_4
[format("rgba16f")]
RWTexture2D<float4> rwt2D_4;

//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = dispatchThreadID.x;
    
    float val = 0;
 
    // Do a format converting write2!
    rwt2D[uint2(idx, idx)] = idx - 1;
    rwt2D_2[uint2(idx, idx)] = float2(idx + idx, idx * idx);
    rwt2D_4[uint2(idx, idx)] = float4(idx + 97, idx + 8, idx + 16, idx + 24);    
    
    // May not be strictly necessary
    AllMemoryBarrierWithGroupSync();
    
    // Do read converting
    // There is *only* CUDA support for half/float *converting* reads for 1d, 2d, 3d shapes for RWTexture/surface
    
    float4 v4 = rwt2D_4[uint2(idx, idx)];
    float2 v2 = rwt2D_2[uint2(idx, idx)];
    float v = rwt2D[uint2(idx, idx)];
    
    val += v4.x + v2.x + v;
    
    outputBuffer[idx] = val;
}
