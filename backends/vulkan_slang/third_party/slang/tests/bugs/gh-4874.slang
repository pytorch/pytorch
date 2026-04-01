// type-legalize-global-with-init.slang
//
// Confirm that type legalization can handle a global constant
// with a resource type or a type that recursively contains
// resources.
//
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -vk
//
//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

//TEST_INPUT:ubuffer(data=[1 2 3 4 5 6 7 8], stride=4):name=inputBuffer
RWStructuredBuffer<uint> inputBuffer;

static const RWStructuredBuffer<uint> gBuffer = inputBuffer;

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

static const Stuff gStuff = Stuff( inputBuffer, inputBuffer );

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
    // CHECK: 321
    // CHECK: 432
    // CHECK: 543
    // CHECK: 654
}