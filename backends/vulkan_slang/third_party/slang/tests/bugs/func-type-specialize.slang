//TEST:SIMPLE(filecheck=CHECK):-target hlsl -entry computeMain -profile cs_6_5 -validate-ir
//TEST_INPUT:ubuffer(data=[0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

// CHECK: void computeMain
// CHECK: RayQuery<0
interface IFoo
{
    int doThing<let f : int>(RayQuery<f> arr);
}


struct HitInfo
{
    static int makeHitInfo<let f : int>(RayQuery<f> q)
    {
        return 0;
    }
}

struct Impl : IFoo
{
    int doThing<let f : int>(RayQuery<f> arr)
    {
        return HitInfo.makeHitInfo(arr);
    }
}

RWStructuredBuffer<int> outVal;
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    Impl impl;
    RayQuery<0> q;
    outVal[dispatchThreadID.x] = impl.doThing(q);
}