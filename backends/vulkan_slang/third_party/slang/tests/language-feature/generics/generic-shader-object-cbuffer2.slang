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
//TEST_INPUT:set gFoo = new Impl<Elem,Elem>{2.0, ubuffer(data=[1.0], stride = 4), ubuffer(data=[2.0], stride = 4)}
//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
//TEST_INPUT:set v = 1.0;

[numthreads(1, 1, 1)]
void computeMain(uniform ParameterBlock<Impl<Elem, Elem>> gFoo,
                 uniform float v,
                 uniform RWStructuredBuffer<float> outputBuffer)
{
    // CHECK: 6.0
    outputBuffer[0] = gFoo.getVal() + v;
}
