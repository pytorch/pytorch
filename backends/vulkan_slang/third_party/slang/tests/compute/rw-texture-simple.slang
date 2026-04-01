//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj
// Doesn't work on DX11 currently - locks up on binding
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -output-using-type -shaderobj
// Produces a different result on DX12 with DXBC than expected(!). So disabled for now
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile cs_6_0 -use-dxil -output-using-type -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -output-using-type -shaderobj

//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -output-using-type -shaderobj

//TEST_INPUT: RWTexture1D(format=R32Float, size=4, content = one, mipMaps = 1):name rwt1D
RWTexture1D<float> rwt1D;
//TEST_INPUT: RWTexture2D(format=R32Float, size=4, content = one, mipMaps = 1):name rwt2D
RWTexture2D<float> rwt2D;
//TEST_INPUT: RWTexture3D(format=R32Float, size=4, content = one, mipMaps = 1):name rwt3D
RWTexture3D<float> rwt3D;

//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = dispatchThreadID.x;
    
    float val = 0.0f;
 
    val += rwt1D[idx];
    val += rwt2D[uint2(idx, idx)];
    val += rwt3D[uint3(idx, idx, idx)];
    
    rwt1D[idx] = idx;
    rwt2D[uint2(idx, idx)] = idx;    
    rwt3D[uint3(idx, idx, idx)] = idx;    
            
    val += rwt1D[idx];
    val += rwt2D[uint2(idx, idx)];
    val += rwt3D[uint3(idx, idx, idx)];
 
    outputBuffer[idx] = val;
}
