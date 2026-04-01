//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IFoo
{
    float sample();
}

//TEST_INPUT: type_conformance Bar:IBar = 0
[anyValueSize(64)]
interface IBar
{
    float sample();
}

//TEST_INPUT: type_conformance Foo:IFoo = 0
struct Foo : IFoo
{
    float3 albedo;
    float sample()
    {
        return albedo.x;
    }
}

struct Bar : IBar
{
    IFoo foo;
    float sample()
    {
        return foo.sample();
    }
}

struct AnyVal
{
    uint data[16];
}

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    int index = dispatchThreadID.x;

    Foo f;
    f.albedo = float3(1.0);
    Bar bar;
    bar.foo = f;

    AnyVal data = reinterpret<AnyVal>(bar);
    IBar dynBar = createDynamicObject<IBar>(0, data);

    outputBuffer[index] = dynBar.sample();
}
