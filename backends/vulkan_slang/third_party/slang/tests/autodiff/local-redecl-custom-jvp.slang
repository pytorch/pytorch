//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

import test_intrinsics;

dpfloat my_pow_jvp(dpfloat x, dpfloat n)
{
    return dpfloat(
        pow(x.p, n.p),
        x.d * n.p * pow(x.p, n.p-1) + n.d * pow(x.p, n.p) * log(x.p));
}

[ForwardDerivative(my_pow_jvp)]
float _pow(float, float);

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
     {
        dpfloat dpa = dpfloat(5.0, 1.0);
        dpfloat dpn = dpfloat(2, 0.0);

        outputBuffer[0] = __fwd_diff(_pow)(dpa, dpn).d;        // Expect: 10.0
        outputBuffer[1] = __fwd_diff(_pow)(
            dpfloat(dpa.p, 0.0),
            dpfloat(dpn.p, 1.0)).d;                     // Expect: 40.23595
    }
}
