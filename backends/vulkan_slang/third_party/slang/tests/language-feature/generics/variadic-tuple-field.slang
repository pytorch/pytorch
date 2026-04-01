//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK): -output-using-type

struct Set<each T>
{
    Tuple<expand each T> data;
    void f(expand each T v){}
    __init(expand each T d) {
        f(d);
        data = makeTuple(d);
    }
}

//TEST_INPUT: set outputBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<float> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    let set = Set<float>(1.0);
    outputBuffer[0] = set.data._0;
    // CHECK: 1.0
}
