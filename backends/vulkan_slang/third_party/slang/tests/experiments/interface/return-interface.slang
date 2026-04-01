//TEST_TEST:SIMPLE:-target dxil -entry computeMain -profile cs_6_2

/* 
Testing returning an interface type.
 
Works.

Uses dynamic dispatch like code - but in the end doesn't require any switches (there is only a single impl). That means it could in principal remove all the extraeous code used assuming it does dynamically dispatch.
 */

interface IGetValue
{
    int getValue(int index);
};

struct SomeType : IGetValue
{
    __init(int inOffset) { offset = inOffset; }
    int getValue(int index) { return index + offset; }
    int offset;
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