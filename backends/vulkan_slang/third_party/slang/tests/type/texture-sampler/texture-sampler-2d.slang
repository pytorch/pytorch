//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT: TextureSampler2D(size=4, content=one):name t2D
Sampler2D<float> t2D;

//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = dispatchThreadID.x;
    float u = idx * (1.0f / 4);
    
    float val = 0.0f;
   
    val += t2D.SampleLevel(float2(u, u), 0);

    outputBuffer[idx] = val;
}
