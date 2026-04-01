//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type

enum MyEnum { A, B, C };

[BackwardDerivative(mDiff)]
float m<let M : MyEnum>(float x) 
{ 
    switch (M)
    {
        case MyEnum.A:
            return x * x;
        case MyEnum.B:
            return x;
        case MyEnum.C:
            return 3 * x;
        default:
            return 0;
    }
}

void mDiff<let M : MyEnum>(inout DifferentialPair<float> x, float dResult)
{
    switch (M)
    {
        case MyEnum.A:
            updateDiff(x, 2 * dResult * x.p);
            break;
        case MyEnum.B:
            updateDiff(x, dResult);
            break;
        case MyEnum.C:
            updateDiff(x, 3 * dResult);
            break;
        default:
            updateDiff(x, 0);
            break;
    }
}

[Differentiable]
float test(float x)
{
    return m<MyEnum.A>(x);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    var a = diffPair(3.0);
    __bwd_diff(test)(a, 1.0);
    outputBuffer[dispatchThreadID.x] = a.d;
    // CHECK: 6.0
}
