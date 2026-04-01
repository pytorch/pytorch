//DISABLE_TEST:SIMPLE:-target dxil -entry computeMain -profile cs_6_2

/* Fails with:

(0): internal error 99999: unimplemented feature in Slang compiler: unexpected IR opcode during code emit
*/

// Dynamic dispatch with a resource type
RWStructuredBuffer<float> outputBuffer;

interface IGetTexture
{
    Texture2D<float> getTexture();
};

struct SomeData : IGetTexture
{
    Texture2D<float> getTexture() { return tex; }
    Texture2D<float> tex;
};

interface IInterface
{
    associatedtype Type;
    IGetTexture getType();
};

// Need public to make these conformances available
export struct A : IInterface
{
    typedef SomeData Type;
    IGetTexture getType() { Type t = { gTexA }; return t; }
};

export struct B : IInterface
{
    typedef SomeData Type;
    IGetTexture getType() { Type t = { gTexB }; return t; }
};

Texture2D<float> gTexA, gTexB;


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;

    A a;
    B b;
    
    IInterface intf;
    
    if (tid & 1)
    {
        intf = a;
    }
    else
    {
        intf = b;
    }
    
    var getTex = intf.getType();
    let tex = getTex.getTexture();
    
	outputBuffer[tid] = tex.Load(int3(tid, tid, 0));
}
