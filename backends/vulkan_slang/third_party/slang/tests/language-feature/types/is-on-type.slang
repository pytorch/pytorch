
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -vk

// Test that `is` operator works on generic type param.

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

interface I
{

}

struct A : I
{
    uint i;
};

struct B : I
{
    float2 f2 = float2(0.0f);
};

func test<T : I>(T t) -> int
{
    int rs = 0;
    if (T is A)
    {
        rs++;
    }
    if (T is B)
    {
        rs+=2;
    }
    return rs;
}

[shader("compute")]
[numthreads(1,1, 1)]
void computeMain()
{
    B b = {};
    // CHECK: 2
    outputBuffer[0] = test(b);
}
