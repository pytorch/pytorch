//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-compute -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain()
{
    // Test that use `row_major` on a local variable with init expr
    // works.
    row_major float4x3 m = float4x3(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
    outputBuffer[0] = m[1][2]; // Expect: 6
    // BUFFER: 6
}