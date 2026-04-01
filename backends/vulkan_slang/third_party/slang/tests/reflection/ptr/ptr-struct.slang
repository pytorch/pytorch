//TEST(64-bit):REFLECTION:-stage compute -no-codegen -target host-callable -entry computeMain

struct AnotherStruct
{
    float a;
    int b;
    Ptr<int> ptrC;
};

struct SomeStruct
{    
    Ptr<int> ptrInt;
    int* ptrInt2;
    int anInt;
    AnotherStruct another;
    Ptr<AnotherStruct> anotherPtr;
};

RWStructuredBuffer<SomeStruct> inputBuffer;

RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    outputBuffer[dispatchThreadID.x] = int(dispatchThreadID.x);
}