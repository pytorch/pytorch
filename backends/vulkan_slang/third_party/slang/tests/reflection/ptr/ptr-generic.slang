//TEST(64-bit):REFLECTION:-stage compute -no-codegen -target host-callable -entry computeMain

struct GenericStruct<T, let N: int>
{    
    T someT;
    int values[N];

    Ptr<GenericStruct<float, 2>> genericPtr;
};

Ptr<GenericStruct<int, 4>> genericPtr;

RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    

    outputBuffer[dispatchThreadID.x] = genericPtr.someT; 
}