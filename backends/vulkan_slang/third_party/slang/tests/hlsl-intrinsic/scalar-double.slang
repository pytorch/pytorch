// It doesn't look like fxc, dxc, vk support double versions of many of the intrinsics, so they are disabled here.

// TODO(JS):
// NOTE! war-double-host-callable category is a work around for issues with x86 & host-callable

//TEST(compute, war-double-host-callable):COMPARE_COMPUTE_EX:-cpu -compute 
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -render-feature double
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);
    
    double f = idx * (1.0f / (4.0f));

    int it = 0;
    double ft = 0.0l;
    
    // fmod
    ft += int(((f % 0.11l) * 100) + 0.5l);
    
    ft += sin(f);
    ft += cos(f);
    ft += tan(f);
    
    ft += asin(f);
    ft += acos(f);
    ft += atan(f);
    
    ft += atan2(f, 2.0l);
    
    {
        double sf, cf;
        sincos(f, sf, cf);
        
        ft += sf;
        ft += cf;
    }
    
    ft += rcp(1.0l + f);
    ft += sign(f - 0.5l);
    
    ft += saturate(f * 4 - 2.0l);
    
    ft += sqrt(f);
    ft += rsqrt(1.0l + f);
           
    ft += exp2(f);
    ft += exp(f);
        
        
    ft += frac(f * 3);
    ft += ceil(f * 5 - 3);
    
    ft += floor(f * 10 - 7);
    ft += trunc(f * 7);
    
    ft += log(f + 10.0l);
    ft += log2(f * 3 + 2);

    {
        double v[] = { 1, 10, 100, 1000 };
        ft += int(log10(v[idx]) + 0.5l);
    }
       
    ft += abs(f * 4 - 2.0l);
           
    ft += min(0.5l, f);
    ft += max(f, 0.75l);

    ft += pow(0.5l, f);

    ft += smoothstep(0.2l, 0.7l, f);
    ft += lerp(-100.0l, 100.0l, f);

    ft += clamp(f, 0.1l, 0.3l);

    ft += step(f, 0.5l);

    {
        uint low, high;
        asuint(f * 2.0l, low, high);
        ft += asdouble(low, high);
    }
    
    ft += ldexp(23.2, f);
    
    outputBuffer[idx] = int(ft * 16);
}