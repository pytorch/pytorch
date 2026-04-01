//TEST(compute):COMPARE_COMPUTE: -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj -output-using-type

struct S
{
    int4 data;
    __subscript(int x, int y) -> int
    {
        [__unsafeForceInlineEarly] get { return data[y * 2 + x]; }
        [__unsafeForceInlineEarly] set { data[y * 2 + x] = newValue;}
    }
}

int test()
{
    S s = {};
    s[1, 0] = 1;
    s[1, 1] = 2;
    let b = s[1, 0] + s[1, 1];
    return b;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    outputBuffer[tid] = test();
}
