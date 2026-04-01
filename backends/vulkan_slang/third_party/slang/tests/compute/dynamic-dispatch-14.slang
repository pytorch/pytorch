// Test using interface typed shader parameters with associated types.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile sm_6_0 -use-dxil
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx11 -profile sm_5_0
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -vk -profile glsl_440
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -cpu



[anyValueSize(8)]
interface IAssoc
{
    int eval();
}

[anyValueSize(8)]
interface IInterface
{
    associatedtype TAssoc : IAssoc;
    TAssoc run(int input);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<int> gOutputBuffer;

//TEST_INPUT: set gCb = new StructuredBuffer<IInterface>{new MyImpl{1}};
RWStructuredBuffer<IInterface> gCb;

//TEST_INPUT: set gCb1 = new StructuredBuffer<IInterface>{new MyImpl{1}}
RWStructuredBuffer<IInterface> gCb1;

[numthreads(4, 1, 1)]
void computeMain(int3       dispatchThreadID : SV_DispatchThreadID)
{
    let tid = dispatchThreadID.x;

    let inputVal : int = tid;
    IInterface v0 = gCb.Load(0);
    IInterface v1 = gCb1[0];
    let outputVal = v0.run(inputVal).eval() + v1.run(inputVal).eval();

    gOutputBuffer[tid] = outputVal;
}

// Type must be marked `public` to ensure it is visible in the generated DLL.
export struct MyImpl : IInterface
{
    int val;
    export struct TAssoc : IAssoc
    {
        int val;
        int eval() { return val; }
    }
    TAssoc run(int input)
    {
        TAssoc rs;
        rs.val = input + val;
        return rs;
    }
};

export struct MyImpl2 : IInterface
{
    int val;
    export struct TAssoc : IAssoc
    {
        int val;
        int eval() { return val; }
    }
    TAssoc run(int input)
    {
        TAssoc rs;
        rs.val = input - val;
        return rs;
    }
};
