//TEST:SIMPLE(filecheck=CHECK): -entry computeMain -profile cs_5_0 -target hlsl

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct G<let m : uint>
{
    static const int r = m + 1;
}

// CHECK: int f_0()
// CHECK: return int(3)

int f<let n : int>(G<n> v)
{
    return v.r;
}
static const int c = (int8_t)255;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    G<2> g;
    int tid = dispatchThreadID.x;
    outputBuffer[tid] = f(g);
}
