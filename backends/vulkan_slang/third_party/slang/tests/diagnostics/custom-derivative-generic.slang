//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;

interface IFoo
{
    static float bar1(float x);

    // CHECK-DAG: {{.*}}(13): error 31152
    [PrimalSubstitute(bar1)]
    static float bar(float x);

    static DifferentialPair<float> dd(DifferentialPair<float> x);
}

__generic<let N:int>
float f(float x)
{
    return N*x*x;
}

// CHECK-DAG: {{.*}}(26): error 31153
[ForwardDerivative(IFoo.dd)]
float bbb(float x);

// CHECK-DAG: {{.*}}(30): error 31152
[ForwardDerivativeOf(IFoo.bar)]
DifferentialPair<float> dd1(DifferentialPair<float> x)
{
    return x;
}

// CHECK-DAG: {{.*}}(37): error 31151
[BackwardDerivativeOf(f)]
DifferentialPair<float> df<let N:int>(inout DifferentialPair<float> x, float dOut)
{
    var primal = x.p * x.p;
    var diff = 2 * x.p * x.d * N;
    return DifferentialPair<float>(primal, diff);
}
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(3.0, 1.0);
        outputBuffer[1] = __fwd_diff(f<3>)(dpa).d; // Expect: 6.0
    }
}
