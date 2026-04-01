//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj

enum MyId : uint {}
extension MyId { uint get() { return (uint)this; } }

int test(MyId id)
{
    if (id.get() == 4)
    {
        return (int)id;
    }
    return 0;
}

__intrinsic_type(UInt)
struct MyId2
{
    __init(uint val) { this = __slang_noop_cast<MyId2>(val); }

    __intrinsic_op(0) int get();
}

int test2(MyId2 id)
{
    if (id.get() == 4)
    {
        return id.get();
    }
    return 0;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    uint inVal = tid;
    uint outVal = test(MyId(4)) + test2(MyId2(4));
    // CHECK: 8
    outputBuffer[tid] = outVal;
}

