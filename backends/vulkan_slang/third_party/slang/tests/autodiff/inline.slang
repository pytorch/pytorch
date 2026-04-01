//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUFFER):-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUFFER):-vk -compute -shaderobj -output-using-type
//TEST:SIMPLE(filecheck=CHECK):-stage compute -entry computeMain -target hlsl

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

// CHECK-NOT: void mySqr{{.*}}(

// Test that calls to a ForceInline function stil get correct custom derivative.
[BackwardDerivative(bwd_mySqr)]
[ForceInline]
void mySqr(float x, out float y)
{
    y = x * x;
}

void bwd_mySqr(inout DifferentialPair<float> dpx, in float.Differential dy)
{
    dpx = DifferentialPair<float>(dpx.p, 1001.0);
}

[Differentiable]
void myF(float x, out float y)
{
    mySqr(x, y);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(2.0, 1.0);
        __bwd_diff(myF)(dpa, 1.0);
        // BUFFER: 1001.0
        outputBuffer[0] = dpa.d;

        float o;
        myF(1.0, o);
        outputBuffer[1] = o;
    }
}
