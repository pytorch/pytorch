//TEST(compute):COMPARE_COMPUTE:-cpu -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(in uint i : SV_GroupIndex)
{
    outputBuffer[i] = int(asdouble(uint(i * 2), uint(i * 2 + 1)) * 100);
}
