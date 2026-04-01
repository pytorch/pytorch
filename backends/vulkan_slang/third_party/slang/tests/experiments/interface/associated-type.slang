//TEST_TEST:SIMPLE:-target dxil -entry computeMain -profile cs_6_2

/* 
Testing how an associated type can be used.

In this example because we get the Type from the interface, it doesn't know what actual type it is.

.slang(43): error 30027: 'getValue' is not a member of 'Type'.
    outputBuffer[dispatchThreadID.x] = x + r.getValue(x);

From an error message point of view this perhaps somewhat confusing because there aren't any methods on Type in this context. 

It would be possible for an implementation to determine intf is impl, and do the right thing, although not in general. 
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
};

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