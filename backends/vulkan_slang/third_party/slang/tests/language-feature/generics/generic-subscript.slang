//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK): -output-using-type

//TEST_INPUT: set output = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<float> output;

struct Tx
{
    float x;
    __subscript<I>(I index) -> float
        where I:IInteger
    {
        get { return x + index.toInt(); }
        set { x = newValue;}
    }
}

[numthreads(1,1,1)]
void computeMain()
{
    Tx obj;
    obj[0] = 3.0;
    // CHECK: 5.0
    output[0] = obj[2];
}