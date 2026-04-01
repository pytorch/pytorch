//DISABLE_TEST:SIMPLE:-target dxil -entry computeMain -profile cs_6_2

/* 

Doesn't work:

.slang(51): error 30027: 'getValue' is not a member of 'Type'.
    outputBuffer[dispatchThreadID.x] = x + r.getValue(x);
                                            ^                 
    */

interface ISomeInterface
{
    int getValue(int index);
};

struct SomeType : ISomeInterface
{
    int getValue(int index) { return data.Load(index + offset); }
    
    int offset;
    ByteAddressBuffer data;     
};

interface IInterface
{
    associatedtype Type;
    
    Type get(int offset);
};

struct InterfaceImpl : IInterface
{
    typedef SomeType Type;
    
    Type get(int offset) 
    {
        Type a;
        a.offset = offset;
        a.data = byteAddressBuffer;
        return a;
    }; 
};

ByteAddressBuffer byteAddressBuffer;
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 4, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int x = dispatchThreadID.x;
    
    InterfaceImpl impl;
    
    IInterface intf = impl;
    
    // Works if we get from impl. Doesn't work if we get from the interface
    // saying that 'Type' doesn't define getValue.
    
    var r = intf.get(16);
    
    outputBuffer[dispatchThreadID.x] = x + r.getValue(x);
}