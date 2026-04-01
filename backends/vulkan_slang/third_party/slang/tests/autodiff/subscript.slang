//TEST:COMPARE_COMPUTE(filecheck-buffer=CHK): -output-using-type

interface ITest
{
    __subscript(int i) -> float
    {
        [BackwardDifferentiable] get;
    }
}
struct Test : ITest
{
    __subscript(int i) -> float
    {
        [BackwardDifferentiable] get { return 5.0f * i; }
    }
}

[Differentiable]
float test(ITest arg)
{
    return arg[1];
}

//TEST_INPUT:set output = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<float> output;

[numthreads(1,1,1)]
void computeMain()
{
    Test t = {};
    output[0] = test(t);
    // CHK: 5.0
}