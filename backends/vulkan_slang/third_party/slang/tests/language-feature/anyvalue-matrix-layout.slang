//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK): -output-using-type

interface IFoo
{
    float getVal();
}

struct Foo : IFoo
{
    column_major float3x2 m;
    float getVal()
    {
        return m[2][0];
    }
}

//TEST_INPUT: type_conformance Foo:IFoo = 0

//TEST_INPUT: set gFoo = ubuffer(data=[0 0 0 0 1.0 2.0 3.0 4.0 5.0 6.0], stride=4)
RWStructuredBuffer<IFoo> gFoo;

//TEST_INPUT: set outputBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<float> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    // CHECK: 3.0
    outputBuffer[0] = gFoo[0].getVal();
}