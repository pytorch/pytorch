// This test is to see what intrinsics are available on VK

//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -output-using-type -shaderobj -render-feature double

// We don't want to run a cuda test here...
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<double> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);
    
    double f = idx * (1.0f / (4.0f));
    
    double ft = 0.0l;
    
    // fmod
    ft += int(((f % 0.11l) * 100) + 0.5l);
    
    ft += rcp(1.0l + f);
    ft += sign(f - 0.5l);
    
    ft += saturate(f * 4 - 2.0l);
    
    ft += sqrt(f);
    ft += rsqrt(1.0l + f);
        
    ft += frac(f * 3);
    ft += ceil(f * 5 - 3);
    
    ft += floor(f * 10 - 7);
    ft += trunc(f * 7);
       
    ft += abs(f * 4 - 2.0l);
           
    ft += min(0.5l, f);
    ft += max(f, 0.75l);

    ft += smoothstep(0.2l, 0.7l, f);
    ft += lerp(-100.0l, 100.0l, f);

    ft += clamp(f, 0.1l, 0.3l);

    ft += step(f, 0.5l);

    {
        uint low, high;
        asuint(f * 2.0l, low, high);
        ft += asdouble(low, high);
    }
    
    outputBuffer[idx] = ft;
}