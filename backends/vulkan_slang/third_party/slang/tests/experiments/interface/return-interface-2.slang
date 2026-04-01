//TEST_TEST:SIMPLE:-target dxil -entry computeMain -profile cs_6_2

/* 
Testing returning an interface type.

Fails with:

.slang(6): error 41011: type 'SomeType' does not fit in the size required by its conforming interface.
struct SomeType : IGetValue
^~~~~~

That's understandable in so far as dynamic dispatch appears to assume anyValue(16) by default. On the other hand, the compiler will know all conforming types (well at linkage anyway), and so potentially could work out an appropriate size.
*/

interface IGetValue
{
    int getValue(int index);
};

struct SomeType : IGetValue
{
    __init(int inOffset) { offset = inOffset; another = float4(inOffset); }
    int getValue(int index) { return index + offset; }
    int offset;
    float4 another;
};

interface IInterface
{
    IGetValue get(int offset);
};

struct InterfaceImpl : IInterface
{   
    IGetValue get(int offset) 
    {
        return SomeType(offset);
    }; 
};

RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 4, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int x = dispatchThreadID.x;
    
    InterfaceImpl impl;
    
    IInterface intf = impl;
       
    var r = intf.get(16);
    
    outputBuffer[dispatchThreadID.x] = x + r.getValue(x);
}