// Native half not supported on CPU currently
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj
// Doesn't work on DX11 currently - locks up on binding
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -output-using-type -shaderobj
// Produces a different result on DX12 with DXBC than expected(!). So disabled for now
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -output-using-type -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile cs_6_0 -use-dxil -output-using-type -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -output-using-type -shaderobj -render-features half

// Note that this test is a little silly. The output does not confirm that the write actually worked.
// half-rw-texture-convert2.slang tests this

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
    
    float val = idx;
 
    // Do a format converting write!
    rwt2D[uint2(idx, idx)] = val;    
    
    rwt2D_2[uint2(idx, idx)] = float2(val * 2, val * 3);    
    
    rwt2D_4[uint2(idx, idx)] = float4(val + 1, val - 1, val * 4, val * -4);    
    
    outputBuffer[idx] = val;
}
