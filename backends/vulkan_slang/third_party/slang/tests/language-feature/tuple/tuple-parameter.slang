// Test using tuples in shader parameters.

//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-dx12 -use-dxil -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -shaderobj

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

//TEST_INPUT: set g_texture = new Tuple<Texture2D, SamplerState>{Texture2D(size=4, content=one), Sampler}
ConstantBuffer<Tuple<Texture2D, SamplerState>> g_texture;
//TEST_INPUT: set g_texture2 = new Tuple<Texture2D, SamplerState>{Texture2D(size=4, content=one), Sampler}
ConstantBuffer<Tuple<Texture2D, SamplerState>> g_texture2;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // CHECK: 4
    outputBuffer[0] = (int)g_texture2._0.SampleLevel(g_texture._1, float2(0.0), 0).x;
}
