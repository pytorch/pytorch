//DISABLE_TEST:SIMPLE: -target hlsl -entry computeMain -stage compute

// This is a simplified version of the mul-crash.slang test, that shows the underlying issue is recursion.
// This test crashes when enabled, with a stack overrun. The crash happens when lowering into IR.
//
// It appears recursion in GLSL/HLSL is not allowed (at least with earlier shader models), but the compiler 
// shouldn't crash. Moreover the Slang language can support recursion today on non GPU targets.

RWStructuredBuffer<int> outputBuffer;

int doRecursive(int a)
{
    if (a > 0)
    {
        return doRecursive(a - 1) + 2;
    }
    return 10;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;

    let val = doRecursive(tid);

    outputBuffer[tid] = val;
} 