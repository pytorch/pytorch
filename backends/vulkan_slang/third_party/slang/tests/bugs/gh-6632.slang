//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu

// CHECK: 40000000

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer

// Non-differentiable function with no_diff parameters and return type
no_diff float targetFunc(no_diff float x)
{
    return x * 2.0f;
}

[Differentiable]
float errorForward(no_diff float x)
{
    float result = targetFunc(x);
    return result;
}

RWStructuredBuffer<float> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float input = 1.0f;

    outputBuffer[0] = errorForward(input);
}

