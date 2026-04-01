//DISABLE_TEST:SIMPLE:-target hlsl -entry computeMain -profile cs_6_2

/* 
The following works, but *requires* the rotateLeft free function to work around the swizzle issue (seen in explicit-specialization.slang).
*/

RWStructuredBuffer<int> outputBuffer;

interface IRotatable 
{
    This rotateLeft();
};

extension int : IRotatable
{
    This rotateLeft() { const uint u = this; return This((u << 1) | (u >> 31)); }
};

extension uint : IRotatable
{
    This rotateLeft() { let u = this; return This((u << 1) | (u >> 31)); }
};

T rotateLeft<T : IRotatable>(T a) { return a.rotateLeft(); }

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;

    let v = rotateLeft(tid) + rotateLeft((int)tid);
   
    outputBuffer[tid] = v;
}

