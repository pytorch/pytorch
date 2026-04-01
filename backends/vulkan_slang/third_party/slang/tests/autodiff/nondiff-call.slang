//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef DifferentialPair<float3> dpfloat3;

[ForwardDifferentiable]
float f(float x)
{
    return x * x + x * x * x;
}

[ForwardDifferentiable]
float f2(float x)
{
    return f(x);
}

float g(float x)
{
    return x * x + x * x * x;
}

[ForwardDifferentiable]
float g2(float x)
{
    return no_diff(g(x));
}

struct A
{
    float o;

    [ForwardDifferentiable]
    float doSomethingDifferentiable(float b)
    {
        return o + b;
    }

    float doSomethingNotDifferentiable(float b)
    {
        return o * b;
    }
}

[ForwardDifferentiable]
float h2(A a, float k)
{
    float v = k * k;
    return no_diff(a.doSomethingNotDifferentiable(k)) + v;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
     {
         outputBuffer[0] = f2(1.0); // Expect: 2.0
         outputBuffer[1] = __fwd_diff(f2)(dpfloat(1.0, 1.0)).d; // Expect: 5.0
         outputBuffer[2] = __fwd_diff(f2)(dpfloat(1.0, 1.0)).p; // Expect: 2.0
         outputBuffer[3] = __fwd_diff(g2)(dpfloat(1.0, 1.0)).d; // Expect: 0.0
         outputBuffer[4] = __fwd_diff(h2)({1.0}, DifferentialPair<float>(1.0, 2.0)).d; // Expect: 4.0
    }
}
