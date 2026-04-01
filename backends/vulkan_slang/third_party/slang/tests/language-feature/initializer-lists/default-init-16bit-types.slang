// simple-namespace.slang

//TEST(compute):COMPARE_COMPUTE:-vk -render-feature int16

// Test that default initialization works with 16-bit types under Vulkan.

struct S
{
    int a;
    int16_t b;
    float16_t c;
    int d;
}

int test(int val)
{
    S s = {};
    s.a += val;
    s.b += int16_t(val*16);
    s.c += float16_t(val*16*16);
    s.d += val*16*16*16;
    return s.a + s.b + int(s.c) + s.d;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = test(inVal);
    outputBuffer[tid] = outVal;
}
