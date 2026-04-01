// Test calling differentiable function through dynamic dispatch.

//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type -compile-arg -skip-spirv-validation -emit-spirv-directly

//TEST_INPUT:ubuffer(data=[2 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IGetter : IDifferentiable
{
    [Differentiable]
    float get(uint id);
}

struct GetterImpl : IGetter
{
    float[8] data;

    __init(float[8] data)
    { this.data = data; }

    [Differentiable]
    float get(uint id)
    {
        return data[id];
    }
}
interface IFoo<int N>
{
    associatedtype Params : IGetter;

    [Differentiable]
    Params bar();
}

[BackwardDerivative(load_bwd)]
[ForwardDerivative(load_fwd)]
float load(uint id)
{
    return outputBuffer[id] + 2;
}

DifferentialPair<float> load_fwd(uint id)
{
    return DifferentialPair<float>(load(id), 3.f);
}

void load_bwd(uint id, float.Differential dOut)
{
    outputBuffer[id + 8] = dOut;
}

struct FooImpl1: IFoo<8>
{   
    typealias Params = GetterImpl;

    __init()
    { }

    [Differentiable]
    Params bar()
    {
        float x = load(0);
        return GetterImpl({x, x+1, x+2, x+3, x+4, x+5, x+6, x+7});
    }
}

/*
// There's a slight issue with dynamic dispatch over generic interfaces. Uncomment after that is fixed.

struct FooImpl2: IFoo<8>
{
    typealias Params = GetterImpl;

    __init()
    { }

    [Differentiable]
    Params bar()
    {
        float x = 2 * load(0);
        return GetterImpl({x, x+5, x+7, x+9, x+11, x+13, x+15, x+17});
    }
}
*/

IFoo<8> getFoo(uint id)
{
    /*if (id == 0)
        return FooImpl1();
    else
        return FooImpl2();*/
    return FooImpl1();
}

[Differentiable]
float doThing(uint id)
{
    IFoo<8> foo = getFoo(id);
    return foo.bar().get(0);
}

[shader("compute")]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer[0] = doThing(0); // CHECK: 2.0
    outputBuffer[1] = doThing(1); // CHECK: 4.0

    outputBuffer[2] = fwd_diff(doThing)(0).d; // CHECK: 3.0
    outputBuffer[3] = fwd_diff(doThing)(1).d; // CHECK: 3.0
}