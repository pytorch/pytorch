//TEST:CPP_COMPILER_COMPILE: -profile cs_5_0 -entry computeMain -target callable

struct Thing
{
    int a;
    float b;
};

static int value;

// Don't use parameter block for now
//ParameterBlock<AnotherThing> s_paramBlock;

ConstantBuffer<Thing> thing3;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out
RWStructuredBuffer<int> outputBuffer;

Texture2D<float> tex;
SamplerState sampler;
    
void doSomething(int a, inout float3 v[2])
{
    v[0] = float3(float(a));
    v[1] = float3(float(a + 1));
}    
    
[numthreads(4, 1, 1)]
void computeMain(
    uint3 dispatchThreadID : SV_DispatchThreadID, 
    uniform Thing thing, 
    uniform Thing thing2)
{
    uint tid = dispatchThreadID.x;
    
    int2 fromScalar = tid.x;
    uint2 another = {};
    
    float2 loc = dispatchThreadID.xy * 0.5f;

    float v = tex.Load(int3(tid, tid, 0));
    float s = tex.Sample(sampler, loc);
    
    // This should promote the 0.0 into a float2, 
    float l = tex.Sample(sampler, 0.0);
    
    float3 m = float(3).xxx;
    
    float3 arr[2] = { float3(3), float3(4) };
    doSomething(int(tid), arr);
    
    outputBuffer[tid] = int(tid * tid) + thing.a + thing3.a + int(v + s) + value + fromScalar.y + int(another.y) + int(m.x) + int(l) + int(arr[0].y); // + thing.a;
}