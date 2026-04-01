//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;

struct Foo : IDifferentiable
{
    float a;
    int b;
}

[PreferCheckpoint]
float k()
{
    return outputBuffer[3] + 1;
}

[Differentiable]
void h(float x, float y, out Foo result)
{
    float p = no_diff k();
    float m = x + y + p;
    float n = x - y;
    float r = m * n + 2 * x * y;

    result = {r, 2};
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float x = 2.0;
    float y = 3.5;
    float dx = 1.0;
    float dy = 0.5;

    dpfloat dresult;
    dpfloat dpx = diffPair(x);
    dpfloat dpy = diffPair(y);
    Foo.Differential dFoo;
    dFoo.a = 1.0;
    bwd_diff(h)(dpx, dpy, dFoo);

    outputBuffer[0] = dpx.d; // CHECK: 12.0
    outputBuffer[1] = dpy.d; // CHECK: -4.0
}