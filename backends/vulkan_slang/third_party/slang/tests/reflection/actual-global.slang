//TEST(64-bit):REFLECTION:-stage compute  -no-codegen -target host-callable -entry computeMain

__global int actualGlobal;
int regularGlobal;

RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    outputBuffer[dispatchThreadID.x] = dispatchThreadID.x + actualGlobal + regularGlobal;
}