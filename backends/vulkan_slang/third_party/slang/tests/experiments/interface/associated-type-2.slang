//DISABLE_TEST:SIMPLE:-target dxil -entry computeMain -profile cs_6_2

/* 
Testing how an associated type can be used.

Here we try and only use Type returned in IInterface with methods on IInterface, since presubably the 
impl knows how to deal with it.

Doesn't work.

.slang(52): error 30019: expected an expression of type 'Type', got 'Type'
    outputBuffer[dispatchThreadID.x] = x + intf.getValue(r, 16);

*/

struct SomeType
{
    int getValue(int index) { return index + offset; }
    
    int offset;
};

interface IInterface
{
    associatedtype Type;
    
    Type get(int offset);
    int getValue(Type type, int index);
};

struct InterfaceImpl : IInterface
{
    typedef SomeType Type;
    
    Type get(int offset) 
    {
        Type v;
        v.offset = offset;
        return v;
    }; 
    int getValue(Type type, int index)
    {
        return type.getValue(index);
    }
};

RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 4, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int x = dispatchThreadID.x;
    
    InterfaceImpl impl;
    
    IInterface intf = impl;
    
    var r = intf.get(16);
    
    outputBuffer[dispatchThreadID.x] = x + intf.getValue(r, 16);
}