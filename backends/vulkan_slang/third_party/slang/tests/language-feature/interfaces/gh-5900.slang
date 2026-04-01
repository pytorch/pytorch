//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK): -output-using-type

interface IFoo
{
    float get();
}

extension<FooType : IFoo> FooType {
    float load()
    {
        return get();
    }
}

struct Foo : IFoo
{
    RWStructuredBuffer<float> buffer;
    int dummy;

    float get() { return buffer[0]; }
}

float bugTest(IFoo t)
{
    return t.load();
}

//TEST_INPUT: set input = new Foo { ubuffer(data=[1.0 0 0 0], stride=4), 0 }
ConstantBuffer<Foo> input;

//TEST_INPUT: set output = out ubuffer(data=[0], stride=4)
uniform RWStructuredBuffer<float> output;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    // CHECK: 1.0
    output[0] = bugTest(input);
}