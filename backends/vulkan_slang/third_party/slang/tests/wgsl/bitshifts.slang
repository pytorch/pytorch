//TEST(compute):COMPARE_COMPUTE:-shaderobj

//TEST_INPUT:ubuffer(data=[3 7 8 10], stride=4):name=inputBuffer
RWStructuredBuffer<int> inputBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    int amount = 1;

    outputBuffer[0] = inputBuffer[0] >> amount;

    int2 v2 = int2(inputBuffer[0], inputBuffer[1]) >> amount;
    outputBuffer[1] = v2[0];
    outputBuffer[2] = v2[1];

    int3 v3 = int3(inputBuffer[0], inputBuffer[1], inputBuffer[2]) >> amount;
    outputBuffer[3] = v3[0];
    outputBuffer[4] = v3[1];
    outputBuffer[5] = v3[2];

    int4 v4 = int4(inputBuffer[0], inputBuffer[1], inputBuffer[2], inputBuffer[3]) >> amount;
    outputBuffer[6] = v4[0];
    outputBuffer[7] = v4[1];
    outputBuffer[8] = v4[2];
    outputBuffer[9] = v4[3];

    outputBuffer[10] = inputBuffer[0] << amount;

    v2 = int2(inputBuffer[0], inputBuffer[1]) << amount;
    outputBuffer[11] = v2[0];
    outputBuffer[12] = v2[1];

    v3 = int3(inputBuffer[0], inputBuffer[1], inputBuffer[2]) << amount;
    outputBuffer[13] = v3[0];
    outputBuffer[14] = v3[1];
    outputBuffer[15] = v3[2];

    v4 = int4(inputBuffer[0], inputBuffer[1], inputBuffer[2], inputBuffer[3]) << amount;
    outputBuffer[16] = v4[0];
    outputBuffer[17] = v4[1];
    outputBuffer[18] = v4[2];
    outputBuffer[19] = v4[3];

    v2 = inputBuffer[0] >> int2(amount);
    outputBuffer[20] = v2[0];
    outputBuffer[21] = v2[1];

    v3 = inputBuffer[1] >> int3(amount);
    outputBuffer[22] = v3[0];
    outputBuffer[23] = v3[1];
    outputBuffer[24] = v3[2];

    v4 = inputBuffer[2] >> int4(amount);
    outputBuffer[25] = v4[0];
    outputBuffer[26] = v4[1];
    outputBuffer[27] = v4[2];
    outputBuffer[28] = v4[3];

    v2 = inputBuffer[0] << int2(amount);
    outputBuffer[29] = v2[0];
    outputBuffer[30] = v2[1];

    v3 = inputBuffer[1] << int3(amount);
    outputBuffer[31] = v3[0];
    outputBuffer[32] = v3[1];
    outputBuffer[33] = v3[2];

    v4 = inputBuffer[2] << int4(amount);
    outputBuffer[34] = v4[0];
    outputBuffer[35] = v4[1];
    outputBuffer[36] = v4[2];
    outputBuffer[37] = v4[3];

    v2 = int2(inputBuffer[0], inputBuffer[1]) >> int2(1, 2);
    outputBuffer[38] = v2[0];
    outputBuffer[39] = v2[1];

    v3 = int3(inputBuffer[0], inputBuffer[1], inputBuffer[2]) >> int3(1, 2, 3);
    outputBuffer[40] = v3[0];
    outputBuffer[41] = v3[1];
    outputBuffer[42] = v3[2];

    v4 = int4(inputBuffer[0], inputBuffer[1], inputBuffer[2], inputBuffer[4]) >> int4(1, 2, 3, 4);
    outputBuffer[43] = v4[0];
    outputBuffer[44] = v4[1];
    outputBuffer[45] = v4[2];
    outputBuffer[46] = v4[3];
}
