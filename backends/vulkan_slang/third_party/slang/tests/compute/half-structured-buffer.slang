//TEST(compute):COMPARE_COMPUTE:-vk -compute -profile cs_6_2 -render-features half -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cuda -compute -render-features half -shaderobj

//Disable on Dx12 for now - because writing to structured buffer produces unexpected results
//TEST_DISABLED(compute):COMPARE_COMPUTE:-dx12 -compute -use-dxil -profile cs_6_2 -render-features half -shaderobj

struct Thing
{
    uint pos;
    float radius;
    half4 color;
};

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=16):out,name outputBuffer
RWStructuredBuffer<Thing> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    
    float v = float(tid);
    
    float base = v* 4.0f;
    
    Thing thing;
    thing.pos = tid;
    thing.color = half4(float4(base, base + 1.0f, base + 2.0f, base + 3.0f));    
    thing.radius = v;
    
    outputBuffer[tid] = thing;
}