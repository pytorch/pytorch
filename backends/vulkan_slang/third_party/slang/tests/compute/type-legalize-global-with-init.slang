// type-legalize-global-with-init.slang
//
// Confirm that type legalization can handle a global constant
// with a resource type or a type that recursively contains
// resources.
//
//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj
//
//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

//TEST_INPUT:ubuffer(data=[1 2 3 4 5 6 7 8], stride=4):name=inputBuffer
RWStructuredBuffer<uint> inputBuffer;

static const RWStructuredBuffer<uint> gBuffer = inputBuffer;

struct Stuff
{
    RWStructuredBuffer<uint> a;
    RWStructuredBuffer<uint> b;
}

static const Stuff gStuff = { inputBuffer, inputBuffer };

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
