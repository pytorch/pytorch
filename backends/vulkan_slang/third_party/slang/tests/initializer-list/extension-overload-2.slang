//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-shaderobj -vk
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-shaderobj

struct Foo
{
    int a;
}

extension Foo
{
    __init(int a)
    {
        this.a = a + 5;
    }
}

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain()
{
    // it will prefer the explicit constructor defined in the extension instead of the synthesized one in the struct
    Foo foo = {1.0f};
    //CHECK: BUFFER: 6
    outputBuffer[0] = foo.a;
}
