//DISABLE_TEST:SIMPLE:-target hlsl -entry computeMain -profile cs_6_2

/* Slang doesn't have explicit template specialization. 
Here's an attempt to use the extension mechanism. 

This doesn't work and produces 9 errors of the form
>  error 30052: invalid swizzle pattern 'rotateLeft' on type 'uint'
> let v = tid.rotateLeft();

Presumably for each letter of 'rotateLeft' after r, thinking it's a swizzle?
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

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;

    uint v = tid.rotateLeft();
   
    outputBuffer[tid] = v;
}
