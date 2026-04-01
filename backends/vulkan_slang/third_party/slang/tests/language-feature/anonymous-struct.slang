//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF): -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;
static const int a = 3;
static const struct
{
    int b;
} globalStruct = { a };

int test(int r)
{
    return r + 1;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    struct
    {
        int member1;
        int unusedMember;
    } tt;
    var tmp = test(1);
    static struct
    {
        int member2;
    } uu = {tmp};
    tt.member1 = test(0);
    // BUF: 1
    outputBuffer[0] = tt.member1;
    // BUF: 2
    outputBuffer[1] = uu.member2;
    // BUF: 3
    outputBuffer[2] = globalStruct.b;
}