
//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -output-using-type
//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -output-using-type

//TEST_INPUT: set outputBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<float> outputBuffer;

struct Matrix
{
    float data[16];

    [mutating]
    void map(IFunc<float, float> f)
    {
        for (int i = 0; i < 16; ++i)
        {
            data[i] = f(data[i]);
        }
    }
}

[numthreads(1,1,1)]
void computeMain()
{
    Matrix m = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    int c = 2;
    m.map((float x) => (float)(x * c));
    outputBuffer[0] = m.data[3];
    // CHECK: 8.0
}