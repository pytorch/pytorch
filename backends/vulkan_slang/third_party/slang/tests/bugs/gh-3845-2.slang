//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-shaderobj -vk -output-using-type

struct Foo {float v;};

namespace foo {
    typedef ::Foo Foo; // unexpected '::', expected identifier; works in dxc/hlsl

    float test(Foo f)
    {
        return f.v;
    }
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(4, 1, 1)]
[shader("compute")]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    Foo f;
    f.v = 1.0;
    // CHECK: 1.0
    outputBuffer[0] = foo.test(f);
}
