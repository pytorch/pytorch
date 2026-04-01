//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef DifferentialPair<float3> dpfloat3;

[ForwardDerivative(pow_jvp)]
float pow_(float x, float n)
{
    return pow<float>(x, n);
}

[ForwardDerivative(max_jvp)]
float max_(float x, float y)
{
    return max<float>(x, y);
}

dpfloat pow_jvp(dpfloat x, dpfloat n)
{
    return dpfloat(
        pow(x.p, n.p),
        x.d * n.p * pow(x.p, n.p-1) + 
            ((n.d != 0.0) ? (n.d * pow(x.p, n.p) * log(x.p)) : 0.0));
}

dpfloat max_jvp(dpfloat x, dpfloat y)
{
    return dpfloat(
        max(x.p, y.p),
        (x.p > y.p) ? x.d : y.d);
}


/* Fresnel Schlick example */
[ForwardDifferentiable]
float3 fresnel(float3 f0, float3 f90, float cosTheta)
{
    return f0 + (f90 - f0) * pow_(max_(1 - cosTheta, 0.0), 5);
}

[ForwardDifferentiable]
float g(float a, float b, float c)
{
    return fresnel(float3(a), float3(b), 2 * c * c).y;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
     {
        float3 f0 = float3(0.2, 0.2, 0.2);
        float3 f90 = float3(0.7, 0.7, 0.7);
        float cosTheta = 0.5;

        float3 d_f0 = float3(0.1, 0.1, 0.1);
        float3 d_f90 = float3(0.9, 0.9, 0.9);
        float d_cosTheta = 1.0;

        outputBuffer[0] = __fwd_diff(fresnel)(
            dpfloat3(f0, d_f0),
            dpfloat3(f90, d_f90),
            dpfloat(cosTheta, d_cosTheta)).d.y; // Expect: -0.031250

        float a = 1.0;
        float b = -0.4;
        float c = 0.5;

        float da = -0.4;
        float db = -1.0;
        float dc = 0.2;

        outputBuffer[1] = __fwd_diff(g)(
            dpfloat(a, da),
            dpfloat(b, db),
            dpfloat(c, dc)).d;                 // Expect: -0.24375

        outputBuffer[2] = g(a, b, c);       // Expect: 0.95625

        outputBuffer[3] = __fwd_diff(g)(
            dpfloat(a, da),
            dpfloat(b, db),
            dpfloat(3.0, dc)).d;               // Expect: -0.4;
    }
}
