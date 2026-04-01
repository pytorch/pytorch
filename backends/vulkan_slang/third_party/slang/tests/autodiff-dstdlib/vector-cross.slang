//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[BackwardDifferentiable]
float3 crossImpl(float3 x, float3 y)
{
    return float3(x.y * y.z - y.y * x.z,
                  x.z * y.x - y.z * x.x,
                  x.x * y.y - y.x * x.y);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        let x = float3(-0.51, 0.74, 0.86);
        let y = float3(1.43, -0.92, 4.36);
        let dOut = float3(3.41, 6.55, 2.39);
        var dpx = diffPair(x);
        var dpy = diffPair(y);
        __bwd_diff(cross)(dpx, dpy, dOut);
        outputBuffer[0] = dpx.d[0]; 
        outputBuffer[1] = dpx.d[1]; 
        outputBuffer[2] = dpx.d[2]; 
        outputBuffer[3] = dpy.d[0]; 
        outputBuffer[4] = dpy.d[1]; 
        outputBuffer[5] = dpy.d[2]; 

        __bwd_diff(crossImpl)(dpx, dpy, dOut);
        outputBuffer[6] = dpx.d[0];
        outputBuffer[7] = dpx.d[1];
        outputBuffer[8] = dpx.d[2];
        outputBuffer[9] = dpy.d[0];
        outputBuffer[10] = dpy.d[1];
        outputBuffer[11] = dpy.d[2];
    }
}
