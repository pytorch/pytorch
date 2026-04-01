//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute  -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12  -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile cs_6_0 -use-dxil -shaderobj
// TODO(JS): Doesn't work on vk currently
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute  -shaderobj

// Doesn't work on CUDA, not clear why yet
//DISABLE_TEST_INPUT: Texture1D(format=R16Float, size=4, content = one, mipMaps=1):name tLoad1D
//Texture1D<float> tLoad1D;

//TEST_INPUT: Texture1D(format=R16Float, size=4, content = one):name t1D
Texture1D<float> t1D;
//TEST_INPUT: Texture2D(format=R16Float, size=4, content = one):name t2D
Texture2D<float> t2D;
//TEST_INPUT: Texture3D(format=R16Float, size=4, content = one):name t3D
Texture3D<float> t3D;
//TEST_INPUT: TextureCube(format=R16Float, size=4, content = one):name tCube
TextureCube<float> tCube;

//TEST_INPUT: Texture1D(format=R16Float, size=4, content = one, arrayLength=2):name t1DArray
Texture1DArray<float> t1DArray;
//TEST_INPUT: Texture2D(format=R16Float, size=4, content = one, arrayLength=2):name t2DArray
Texture2DArray<float> t2DArray;
//TEST_INPUT: TextureCube(format=R16Float, size=4, content = one, arrayLength=2):name tCubeArray
TextureCubeArray<float> tCubeArray;

//TEST_INPUT: Sampler:name samplerState
SamplerState samplerState;

//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    float u = idx * (1.0f / 4);
    
    float val = 0.0f;
   
    val += t1D.SampleLevel(samplerState, u, 0); 
    val += t2D.SampleLevel(samplerState, float2(u, u), 0);
    val += t3D.SampleLevel(samplerState, float3(u, u, u), 0);
    val += tCube.SampleLevel(samplerState, normalize(float3(u, 1 - u, u)), 0);
 
    val += t1DArray.SampleLevel(samplerState, float2(u, 0), 0);
    val += t2DArray.SampleLevel(samplerState, float3(u, u, 0), 0);
    val += tCubeArray.SampleLevel(samplerState, float4(u, u, u, 0), 0);
 
    outputBuffer[idx] = val;
}
