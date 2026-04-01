//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;

struct A : IDifferentiable
{
    float x;
    no_diff float y;
}

[BackwardDifferentiable]
float f(A obj)
{
    return obj.y * obj.x * obj.x + obj.y * obj.y;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    A a = {2.0, 3.0};
    var p = diffPair(a);
    let rs = __bwd_diff(f)(p, 1.0);
    outputBuffer[0] = p.d.x;
}
