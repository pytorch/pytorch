//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK): -dx12  -use-dxil -profile cs_6_1 -output-using-type
//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -output-using-type

interface IFoo
{
    float getVal();
    uint64_t getPtrVal();
}

struct Foo : IFoo
{
    column_major float3x2 m;
    int x;
    uint64_t ptr;
    float getVal()
    {
        return m[2][0];
    }
    uint64_t getPtrVal()
    {
        return ptr;
    }
}

//TEST_INPUT: type_conformance Foo:IFoo = 0

//TEST_INPUT: set gFoo = ubuffer(data=[0 0 0 0 1.0 2.0 3.0 4.0 5.0 6.0 0 0 1 2], stride=4)
RWStructuredBuffer<IFoo> gFoo;

//TEST_INPUT: set outputBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<float> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    // CHECK: 3.0
    outputBuffer[0] = gFoo[0].getVal();

    // CHECK: 1.0
    outputBuffer[1] = gFoo[0].getPtrVal()&0xFFFFFFFF;

    // CHECK: 2.0
    outputBuffer[2] = gFoo[0].getPtrVal()>>32;
}