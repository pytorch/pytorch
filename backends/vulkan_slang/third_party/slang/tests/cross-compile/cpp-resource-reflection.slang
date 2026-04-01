//TEST:CPU_REFLECTION: -profile cs_5_0 -entry computeMain -target cpp -no-codegen


struct Thing
{
    int a;
    float b;
    float c;
};

static int value;

// Don't use parameter block for now
//ParameterBlock<AnotherThing> s_paramBlock;

ConstantBuffer<Thing> thing3;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out
RWStructuredBuffer<int> outputBuffer;

Texture2D<float> tex;
SamplerState sampler;
    
[numthreads(4, 1, 1)]
void computeMain(
    uint3 dispatchThreadID : SV_DispatchThreadID, 
    uniform Thing thing, 
    uniform Thing thing2)
{
    uint tid = dispatchThreadID.x;

    // TODO(JS): Doesn't emit correctly on c++ becomes...
    // Vector<float, 2> loc_0 = Vector<float, 2>{Vector<uint32_t, 2>{dispatchThreadID_0.x, dispatchThreadID_0.y}} * 0.50000000000000000000f;
    //float2 loc = dispatchThreadID.xy * 0.5f;

    float2 loc = float2(dispatchThreadID.x * 0.5f, dispatchThreadID.y * 0.5f);

    float v = tex.Load(int3(tid, tid, 0));
    float s = tex.Sample(sampler, loc);
   
    outputBuffer[tid] = int(tid * tid) + thing.a + thing3.a + int(v + s) + value; // + thing.a;
}