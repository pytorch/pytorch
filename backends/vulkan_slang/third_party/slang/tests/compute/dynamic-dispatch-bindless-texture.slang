// Test using interface typed shader parameters with texture typed fields.
//TEST(compute):COMPARE_COMPUTE:-cpu
//TEST(compute):COMPARE_COMPUTE:-cuda

[anyValueSize(16)]
interface IInterface
{
    float run();
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<uint> gOutputBuffer;
//TEST_INPUT: set gCb = new StructuredBuffer<IInterface>{new MyImpl{Texture2D(size=8, content = one), Sampler}}
StructuredBuffer<IInterface> gCb;

[numthreads(4, 1, 1)]
void computeMain(int3       dispatchThreadID : SV_DispatchThreadID)
{
    let tid = dispatchThreadID.x;

    let inputVal : int = tid;
    IInterface v0 = gCb.Load(0);
    SamplerState sampler;
    let outputVal = v0.run();
    gOutputBuffer[tid] = uint(trunc(outputVal));
}


// Type must be marked `public` to ensure it is visible in the generated DLL.
export struct MyImpl : IInterface
{
    Texture2D tex;
    SamplerState sampler;
    float run()
    {
        return tex.Sample(sampler, float2(0.0, 0.0)).x;
    }
};
