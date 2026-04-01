//DISABLE_TEST:SIMPLE:-target hlsl -entry computeMain -profile cs_6_2

/* The docs claim that for arrays we can use the 'alternate' style 

https://github.com/shader-slang/slang/blob/master/docs/language-reference/04-types.md

> int[10] a;

But that produces an error:

.slang(18): error 20001: unexpected '[', expected identifier
    int[4] a = { 1, 2, 3, 4 };
*/

RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 4, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int x = dispatchThreadID.x;

    int[4] a = { 1, 4, 2, 1 };
   
    outputBuffer[x] = a[x];
}
