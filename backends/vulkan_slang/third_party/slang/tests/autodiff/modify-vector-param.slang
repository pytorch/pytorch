//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef float.Differential dfloat;

struct Params
{
    float vv;
}

[Differentiable]
float3 f(float3 x, uint2 v, Params p)
{
    v.x = 1 + int(p.vv);

    x.y = x.x * x.x;
    return x;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        var dpa = diffPair(float3(4.0, 2.0, 3.0));
        Params p;
        p.vv = 0.0;
        __bwd_diff(f)(dpa, uint2(1,2), p, float3(1.0,1.0,1.0));

        // CHECK: 9.0
        outputBuffer[0] = dpa.d.x; // Expect: 9.0
    }
}
