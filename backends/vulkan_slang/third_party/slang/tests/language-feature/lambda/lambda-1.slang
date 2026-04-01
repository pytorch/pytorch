
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

struct Applier
{
    float c;
    void apply(inout Matrix m)
    {
        m.map((float x) =>
        {
            if (x > 2.5)
                return x * c;
            else
                return x + c;
        });
    }
}

[numthreads(1,1,1)]
void computeMain()
{
    Matrix m = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    Applier applier = { 1.5 };
    applier.apply(m);
    outputBuffer[0] = m.data[1];
    // CHECK: 3.5
    outputBuffer[1] = m.data[3];
    // CHECK: 6.0
}