//TEST(compute):COMPARE_COMPUTE_EX:-vk -compute  -output-using-type

//TEST_INPUT: set outputBuffer = out ubuffer(data=[0 0 0 0 0], stride=4);
RWStructuredBuffer<int> outputBuffer;

interface Base
{
    int getValue();
    int getValue(float a);
}

struct Impl1 : Base
{
    // This is static and allowed to implement interface's non-static method.
    static int getValue() { return 5; }
    int getValue(float a) { return int(a) + 5; }
}

struct Impl2 : Base
{
    // This is static with one default parameter and allowed to implement interface's non-static method.
    static int getValue(int a = 3) { return a + 5; }
    int getValue(float a) { return int(a) + 5; }
}

int callGet<T : Base>(T t)
{
    return t.getValue();
}

[numthreads(1, 1, 1)]
void computeMain()
{
    Impl1 impl1;
    Impl2 impl2;

    uint index = 0;

    outputBuffer[index++] = Impl1::getValue();
    outputBuffer[index++] = callGet(impl1);

    outputBuffer[index++] = Impl2::getValue();
    outputBuffer[index++] = callGet(impl2);
    outputBuffer[index++] = impl2.getValue(5);
}
