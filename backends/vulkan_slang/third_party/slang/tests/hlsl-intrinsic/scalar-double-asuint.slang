//TEST(compute):COMPARE_COMPUTE:-cpu -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(in uint i : SV_GroupIndex)
{
    uint low, high;
    asuint(double(i), low, high);
    outputBuffer[i * 2] = int(low);
    outputBuffer[i * 2 + 1] = int(high);
}
