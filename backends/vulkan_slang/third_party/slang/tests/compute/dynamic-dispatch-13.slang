// Test using interface typed shader parameters wrapped inside a `StructuredBuffer`.

//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj
//TEST(compute):COMPARE_COMPUTE:-dx11
//TEST(compute):COMPARE_COMPUTE:-vk
//TEST(compute):COMPARE_COMPUTE:-cuda -shaderobj

[anyValueSize(8)]
interface IInterface
{
    int run(int input);
}

// Specialize gCb1, but not gCb2

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<int> gOutputBuffer;
//TEST_INPUT: set gCb = new StructuredBuffer<IInterface>{new MyImpl{1}};
RWStructuredBuffer<IInterface> gCb;
//TEST_INPUT: set gCb1 = new StructuredBuffer<IInterface>{new MyImpl{1}, new MyImpl2{2}};
RWStructuredBuffer<IInterface> gCb1;

[numthreads(4, 1, 1)]
void computeMain(int3       dispatchThreadID : SV_DispatchThreadID)
{
    let tid = dispatchThreadID.x;

    let inputVal : int = tid;
    IInterface v0 = gCb.Load(0);
    IInterface v1 = gCb1[0];
    let outputVal = v0.run(inputVal) + v1.run(inputVal);

    gOutputBuffer[tid] = outputVal;
}

// Type must be marked `public` to ensure it is visible in the generated DLL.
export struct MyImpl : IInterface
{
    int val;
    int run(int input)
    {
        return input + val;
    }
};
export struct MyImpl2 : IInterface
{
    int val;
    int run(int input)
    {
        return input - val;
    }
};
