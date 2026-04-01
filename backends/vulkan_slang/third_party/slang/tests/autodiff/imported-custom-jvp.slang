//TEST_IGNORE_FILE:

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

import test_intrinsics_jvp;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

[ForwardDifferentiable]
float f(float x)
{
    return pow_(x, 2.0);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(2.0, 1.0);
        dpfloat dpb = dpfloat(1.5, 1.0);

        outputBuffer[0] = __fwd_diff(f)(dpa).d;                      // Expect: 2
    }
}
