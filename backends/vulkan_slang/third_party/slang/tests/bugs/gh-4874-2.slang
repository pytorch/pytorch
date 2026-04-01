// Same to gh-4874.slang, but defining the global resource typed variables
// without the "const" modifier. This should lead to a compile time error.
//
//TEST:SIMPLE(filecheck=CHECK): -target spirv
RWStructuredBuffer<uint> outputBuffer;

RWStructuredBuffer<uint> inputBuffer;

// CHECK: ([[# @LINE+1]]): error 30076:
static RWStructuredBuffer<uint> gBuffer = inputBuffer;

struct Stuff
{
    __init(RWStructuredBuffer<uint> a1, RWStructuredBuffer<uint> b2)
    {
        a = a1;
        b = b2;
    }

    RWStructuredBuffer<uint> a;
    RWStructuredBuffer<uint> b;
}

// CHECK: ([[# @LINE+1]]): error 30076:
static Stuff gStuff = Stuff( inputBuffer, inputBuffer );

uint test(uint x)
{
    return gBuffer[x]
        + gStuff.a[x + 1] * 16
        + gStuff.b[x + 2] * 256;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    let tid = dispatchThreadID.x;
    let inVal = tid;
    let outVal = test(inVal);
    outputBuffer[tid] = outVal;
}