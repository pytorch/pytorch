//TEST(compute):COMPARE_COMPUTE:-shaderobj

//TEST_INPUT:ubuffer(data=[3 7 8], stride=4):name=inputBuffer
RWStructuredBuffer<uint> inputBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    uint a = inputBuffer[0];
    uint b = inputBuffer[1];
    uint c = inputBuffer[2];

    outputBuffer[0] = a+b ^ c;
    outputBuffer[1] = a ^ b+c;

    outputBuffer[2] = a-b | c;
    outputBuffer[3] = a | b-c;

    outputBuffer[4] = a+b & c;
    outputBuffer[5] = a & b+c;

    outputBuffer[6] = a<<b ^ c;
    outputBuffer[7] = a ^ b<<c;

    outputBuffer[8] = a>>b | c;
    outputBuffer[9] = a | b>>c;

    outputBuffer[10] = a<<b & c;
    outputBuffer[11] = a & b<<c;

    outputBuffer[12] = a<b && b!=c;
    outputBuffer[13] = a==b || b>=c;

    outputBuffer[14] = a*b ^ c;
    outputBuffer[15] = a ^ b*c;

    outputBuffer[16] = a/b | c;
    outputBuffer[17] = a | b/c;

    outputBuffer[18] = a*b & c;
    outputBuffer[19] = a & b*c;
}
