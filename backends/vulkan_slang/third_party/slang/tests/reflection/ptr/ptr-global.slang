//TEST(64-bit):REFLECTION:-stage compute -no-codegen -target host-callable -entry computeMain

struct SomeStruct
{    
    Ptr<int> regularGlobal;
    int* regularGlobal2;
    int regularGlobal3;
};

RWStructuredBuffer<SomeStruct> inputBuffer;

RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    outputBuffer[dispatchThreadID.x] = int(dispatchThreadID.x);
}