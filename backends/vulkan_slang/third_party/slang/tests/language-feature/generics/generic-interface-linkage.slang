
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IGetter
{
    float get(uint id);
}

struct GetterImpl : IGetter
{
    float[8] data;

    __init(float[8] data)
    { this.data = data; }

    float get(uint id)
    {
        return data[id];
    }
}
interface IFoo<int N>
{
    associatedtype Params : IGetter;

    Params bar();
}

struct FooImpl1: IFoo<8>
{   
    typealias Params = GetterImpl;

    __init()
    { }

    Params bar()
    {
        float x = outputBuffer[0];
        return GetterImpl({x, x+1, x+2, x+3, x+4, x+5, x+6, x+7});
    }
}

struct FooImpl2: IFoo<8>
{
    typealias Params = GetterImpl;

    __init()
    { }

    Params bar()
    {
        float x = 2 * outputBuffer[0];
        return GetterImpl({x+3, x+5, x+7, x+9, x+11, x+13, x+15, x+17});
    }
}

IFoo<8> getFoo(uint id)
{
    if (id == 0)
        return FooImpl1();
    else
        return FooImpl2();
}

float doThing(uint id)
{
    IFoo<8> foo = getFoo(id);
    return foo.bar().get(0);
}

[shader("compute")]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // CHECK: 0
    outputBuffer[0] = doThing(0);
	// CHECK: 3
    outputBuffer[1] = doThing(1);
}