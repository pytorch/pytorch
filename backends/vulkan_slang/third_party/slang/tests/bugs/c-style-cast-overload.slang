// https://github.com/shader-slang/slang/issues/2189
// See also ./c-style-cast-coerce.slang which represents the essence of this bug

//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE: -shaderobj -vulkan

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct S {};

int f(S)
{
    return 1;
}

int f(float)
{
    return 2;
}

[shader("compute")]
[numthreads(1,1,1)]
void computeMain()
{
    outputBuffer[0] = f(float(0));
    outputBuffer[1] = f((float)1);
    outputBuffer[2] = f((float)0);
    outputBuffer[3] = f((S)0);
}
