// optional.slang

// Test that `is` and `as` operator works on generic types.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -vk -compute -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -cpu -compute -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

T compute<T>(T a1, T a2)
{
    if (a1 is float)
    {
        return reinterpret<T>((a1 as float).value + (a2 as float).value);
    }
    else if (T is int)
    {
        return reinterpret<T>((a1 as int).value - (a2 as int).value);
    }
    return T();
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    // CHECK: 3.0
    outputBuffer[0] = compute(1.0f, 2.0f);

    // CHECK: 1.0
    outputBuffer[1] = compute(2, 1);
}
