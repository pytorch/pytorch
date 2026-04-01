//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute  -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12  -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile cs_6_0 -use-dxil -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type -render-feature hardware-device
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute  -shaderobj -output-using-type

// Doesn't work on CUDA, not clear why yet
//DISABLE_TEST_INPUT: Texture1D(format=R_Float32, size=4, content = one, mipMaps=1):name tLoad1D
//Texture1D<float> tLoad1D;

// Not supported in WGSL: 1D array texture not supported in WGSL
//DISABLE_TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -wgpu

//TEST_INPUT: Texture1D(size=4, content = one):name t1D
Texture1D<float> t1D;
//TEST_INPUT: Texture2D(size=4, content = one):name t2D
Texture2D<float> t2D;
//TEST_INPUT: Texture3D(size=4, content = one):name t3D
Texture3D<float> t3D;
//TEST_INPUT: TextureCube(size=4, content = one):name tCube
TextureCube<float> tCube;

//TEST_INPUT: Texture1D(size=4, content = one, arrayLength=2):name t1DArray
Texture1DArray<float> t1DArray;
//TEST_INPUT: Texture2D(size=4, content = one, arrayLength=2):name t2DArray
Texture2DArray<float> t2DArray;
//TEST_INPUT: TextureCube(size=4, content = one, arrayLength=2):name tCubeArray
TextureCubeArray<float> tCubeArray;

//TEST_INPUT: Sampler:name samplerState
SamplerState samplerState;

//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = dispatchThreadID.x;
    float u = idx * (1.0f / 4);
    
    float val = 0.0f;
   
    val += t1D.SampleLevel(samplerState, u, 0);
    val += t2D.SampleLevel(samplerState, float2(u, u), 0);
    val += t3D.SampleLevel(samplerState, float3(u, u, u), 0);
    val += tCube.SampleLevel(samplerState, normalize(float3(u, 1 - u, u)), 0);
 
    val += t1DArray.SampleLevel(samplerState, float2(u, 0), 0);
    val += t2DArray.SampleLevel(samplerState, float3(u, u, 0), 0);
    val += tCubeArray.SampleLevel(samplerState, float4(u, u, u, 0), 0);
 
    //val += tLoad1D.Load(int2(idx, 0));
 
    outputBuffer[idx] = val;
}
