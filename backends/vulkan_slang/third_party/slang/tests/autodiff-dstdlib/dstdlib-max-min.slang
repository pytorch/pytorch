//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef DifferentialPair<float2> dpfloat2;

[BackwardDifferentiable]
float diffMax(float x, float y)
{
    return max(x, y);
}

[BackwardDifferentiable]
float2 diffMax(float2 x, float2 y)
{
    return max(x, y);
}

[BackwardDifferentiable]
float diffMin(float x, float y)
{
    return min(x, y);
}

[BackwardDifferentiable]
float2 diffMin(float2 x, float2 y)
{
    return min(x, y);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    // Test max() with x < y
    {
        dpfloat dpx = dpfloat(2.0, 1.0);
        dpfloat dpy = dpfloat(5.0, -2.0);
        dpfloat res = __fwd_diff(diffMax)(dpx, dpy);
        outputBuffer[0] = res.p;        // Expect: 5.000000
        outputBuffer[1] = res.d;        // Expect: -2.000000
    }

    // Test max() with x == y
    {
        dpfloat dpx = dpfloat(3.0, 1.0);
        dpfloat dpy = dpfloat(3.0, -2.0);
        dpfloat res = __fwd_diff(diffMax)(dpx, dpy);
        outputBuffer[2] = res.p;        // Expect: 3.000000
        outputBuffer[3] = res.d;        // Expect: -0.500000 (average of 1.0 and -2.0)
    }

    // Test min() with x > y
    {
        dpfloat dpx = dpfloat(5.0, 1.0);
        dpfloat dpy = dpfloat(2.0, -2.0);
        dpfloat res = __fwd_diff(diffMin)(dpx, dpy);
        outputBuffer[4] = res.p;        // Expect: 2.000000
        outputBuffer[5] = res.d;        // Expect: -2.000000
    }

    // Test min() with x == y
    {
        dpfloat dpx = dpfloat(3.0, 1.0);
        dpfloat dpy = dpfloat(3.0, -2.0);
        dpfloat res = __fwd_diff(diffMin)(dpx, dpy);
        outputBuffer[6] = res.p;        // Expect: 3.000000
        outputBuffer[7] = res.d;        // Expect: -0.500000 (average of 1.0 and -2.0)
    }

    // Test backward-mode max() with x == y
    {
        dpfloat dpx = dpfloat(3.0, 0.0);
        dpfloat dpy = dpfloat(3.0, 0.0);
        __bwd_diff(diffMax)(dpx, dpy, 1.0);
        outputBuffer[8] = dpx.d;        // Expect: 0.500000 (half of gradient)
        outputBuffer[9] = dpy.d;        // Expect: 0.500000 (half of gradient)
    }

    // Test backward-mode min() with x == y
    {
        dpfloat dpx = dpfloat(3.0, 0.0);
        dpfloat dpy = dpfloat(3.0, 0.0);
        __bwd_diff(diffMin)(dpx, dpy, 1.0);
        outputBuffer[10] = dpx.d;       // Expect: 0.500000 (half of gradient)
        outputBuffer[11] = dpy.d;       // Expect: 0.500000 (half of gradient)
    }

    // Test vector max() with x == y
    {
        dpfloat2 dpx = dpfloat2(float2(3.0, 4.0), float2(1.0, 2.0));
        dpfloat2 dpy = dpfloat2(float2(3.0, 2.0), float2(-2.0, -3.0));
        dpfloat2 res = __fwd_diff(diffMax)(dpx, dpy);
        outputBuffer[12] = res.p[0];    // Expect: 3.000000
        outputBuffer[13] = res.d[0];    // Expect: -0.500000 (average of 1.0 and -2.0)
        outputBuffer[14] = res.p[1];    // Expect: 4.000000
        outputBuffer[15] = res.d[1];    // Expect: 2.000000
    }

    // Test vector min() with x == y
    {
        dpfloat2 dpx = dpfloat2(float2(3.0, 4.0), float2(1.0, 2.0));
        dpfloat2 dpy = dpfloat2(float2(3.0, 2.0), float2(-2.0, -3.0));
        dpfloat2 res = __fwd_diff(diffMin)(dpx, dpy);
        outputBuffer[16] = res.p[0];    // Expect: 3.000000
        outputBuffer[17] = res.d[0];    // Expect: -0.500000 (average of 1.0 and -2.0)
        outputBuffer[18] = res.p[1];    // Expect: 2.000000
        outputBuffer[19] = res.d[1];    // Expect: -3.000000
    }
}
