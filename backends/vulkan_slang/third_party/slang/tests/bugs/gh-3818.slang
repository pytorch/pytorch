//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-shaderobj -output-using-type

interface IOperation
{
    [mutating]
    void Store(float v);

    float Load();
};

struct TOperationMax : IOperation
{
    float result;

    [mutating]
    void Store(float v)
    {
        result = v;
    }

    float Load()
    {
        return result;
    }
};

float Run(inout IOperation op, float val)
{
    op.Store(val);
    return op.Load();
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(4, 1, 1)]
[shader("compute")]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TOperationMax opMax;
    opMax.result = 0.f;
    float val = 2.f;
    IOperation op = opMax;
    float result = Run(op, val);
    // CHECK: 2.0
    outputBuffer[0] = result;
}
