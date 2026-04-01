//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -d3d11 -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -d3d12 -shaderobj -output-using-type -use-dxil
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -cpu -shaderobj -output-using-type

interface IFoo
{
    float getVal();
}
interface IElement
{
    float getElementVal();
}

struct Elem : IElement
{
    float x;
    float getElementVal()
    {
        return x;
    }
}

struct Impl<T1:IElement, T2:IElement> : IFoo
{
    float v1;
    RWStructuredBuffer<T1> buffer0;
    RWStructuredBuffer<T2> buffer1;

    float getVal()
    {
        return buffer0[0].getElementVal() + buffer1[0].getElementVal() + v1;
    }
}

//TEST_INPUT:set cparams = new Params { new Impl<Elem,Elem>{2.0, ubuffer(data=[1.0], stride = 4), ubuffer(data=[2.0], stride = 4)}, 1.0, ubuffer(data=[1.0], stride=4) }
//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer

struct Params
{
    ParameterBlock<Impl<Elem, Elem>> gFoo;
    float v;
    RWStructuredBuffer<float> buffer;
}

ConstantBuffer<Params> cparams;
RWStructuredBuffer<float> outputBuffer;
[numthreads(1, 1, 1)]
void computeMain()
{
    // CHECK: 7.0
    outputBuffer[0] = cparams.gFoo.getVal() + cparams.v + cparams.buffer[0];
}
