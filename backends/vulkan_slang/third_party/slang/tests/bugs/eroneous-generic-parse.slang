//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target hlsl -entry computeMain -stage compute

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=16):out
RWStructuredBuffer<int> outputBuffer;

// CHECK: error 20001
// Previously this definition would lead to an infinite loop in parsing.
int doThing<1>() { return 2; }

[numthreads(4, 4, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = dispatchThreadID.x;
   
    outputBuffer[index] = index;
}