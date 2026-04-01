//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef DifferentialPair<float2> dpfloat2;
typedef DifferentialPair<float3> dpfloat3;

[BackwardDifferentiable]
float diffAtan2(float x, float y)
{
    return atan2(x, y);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    float eps = 1e-4;
    {
        dpfloat dpx = dpfloat(5.0, 1.0);
        dpfloat dpy = dpfloat(3.0, 0.0);
        dpfloat res = __fwd_diff(diffAtan2)(dpx, dpy);
        outputBuffer[0] = res.d;
    }

    {
        dpfloat dpx = dpfloat(5.0, 0.0);
        dpfloat dpy = dpfloat(3.0, 1.0);
        dpfloat res = __fwd_diff(diffAtan2)(dpx, dpy);
        outputBuffer[1] = res.d;
    }

    // Test the other 3 quadrants
    {
        dpfloat dpx = dpfloat(-5.0, 1.0);
        dpfloat dpy = dpfloat(3.0, 0.0);
        dpfloat res = __fwd_diff(diffAtan2)(dpx, dpy);
        outputBuffer[2] = res.d;
    }

    {
        dpfloat dpx = dpfloat(-5.0, 0.0);
        dpfloat dpy = dpfloat(3.0, 1.0);
        dpfloat res = __fwd_diff(diffAtan2)(dpx, dpy);
        outputBuffer[3] = res.d;
    }

    {
        dpfloat dpx = dpfloat(-5.0, 1.0);
        dpfloat dpy = dpfloat(-3.0, 0.0);
        dpfloat res = __fwd_diff(diffAtan2)(dpx, dpy);
        outputBuffer[4] = res.d;
    }

    {
        dpfloat dpx = dpfloat(-5.0, 0.0);
        dpfloat dpy = dpfloat(-3.0, 1.0);
        dpfloat res = __fwd_diff(diffAtan2)(dpx, dpy);
        outputBuffer[5] = res.d;
    }

    {
        dpfloat dpx = dpfloat(5.0, 1.0);
        dpfloat dpy = dpfloat(-3.0, 0.0);
        dpfloat res = __fwd_diff(diffAtan2)(dpx, dpy);
        outputBuffer[6] = res.d;
    }

    {
        dpfloat dpx = dpfloat(5.0, 0.0);
        dpfloat dpy = dpfloat(-3.0, 1.0);
        dpfloat res = __fwd_diff(diffAtan2)(dpx, dpy);
        outputBuffer[7] = res.d;
    }

    {
        dpfloat dpx = diffPair(5.0);
        dpfloat dpy = diffPair(3.0);
        __bwd_diff(diffAtan2)(dpx, dpy, 1.0);
        outputBuffer[8] = dpx.d; // Should be equal to outputBuffer[0]
        outputBuffer[9] = dpy.d; // Should be equal to outputBuffer[2]
    }
}
