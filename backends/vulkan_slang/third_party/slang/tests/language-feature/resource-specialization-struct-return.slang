//TEST:SIMPLE(filecheck=CHECK_DXIL):-target dxil -entry computeMain -profile cs_6_2
//CHECK_DXIL: computeMain

// This test demonstrates returning struct with resource.

RWTexture1D<int> g_t;
RWStructuredBuffer<int> outputBuffer;

struct Thing
{
    int a;
    RWTexture1D<int> t;
};

Thing makeThing()
{
 	Thing t;
    t.a = 10;
    t.t = g_t;
    return t;
}

[numthreads(4, 4, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int x = dispatchThreadID.x;
    Thing thing = makeThing();
    outputBuffer[dispatchThreadID.x] = x + thing.t.Load(1); 
}