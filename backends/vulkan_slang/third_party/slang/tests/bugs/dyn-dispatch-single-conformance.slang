//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK): -output-using-type

interface IFoo
{
    float3 get();
}

struct Foo : IFoo
{
    float3 val;
    float3 get() { return val; }
};

//TEST_INPUT: type_conformance Foo:IFoo=0

//TEST_INPUT:set foo = ubuffer(data=[0 0 0 0 1.0 2.0 3.0 0.0], stride=4)
StructuredBuffer<IFoo> foo;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    // CHECK: 1.0
    outputBuffer[0] = foo[0].get().x;
}