//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK): -output-using-type

// Regression test for bug https://github.com/shader-slang/slang/issues/5808

// Using a type defined from a different module
// in a for loop should work.

import lib;

//TEST_INPUT:set output = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<i32> output;

[numthreads(1,1,1)]
void computeMain()
{
    // CHECK: 0
    // CHECK: 1
    // CHECK: 2
    // CHECK: 3
    for (i32 i = 0; i < 4; i++)
    {
        output[i] = i;
    }
}