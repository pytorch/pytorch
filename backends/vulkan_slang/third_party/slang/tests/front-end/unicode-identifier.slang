//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type -xslang -O0

// Test that we can use unicode characters in identifier names.

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

static const float π = float.getPi();

float /*комментарий*/ 圆周率_円周率_원주율()
{
    return π;   
}

[numthreads(1, 1, 1)]
void computeMain()
{
    // CHECK: 3.14
    outputBuffer[0] = 圆周率_円周率_원주율();
}
