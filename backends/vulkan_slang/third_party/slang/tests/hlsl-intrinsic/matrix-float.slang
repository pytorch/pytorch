// TODO(JS):
// NOTE we can't test on VK/gl at the moment because we don't support intrinsics over matrices on that target currently

//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -output-using-type -shaderobj -render-feature hardware-device
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -output-using-type -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-cuda -compute -output-using-type -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl
// Not supported in WGSL: Integer matrices, Double and other unsupported scalar types
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-wgpu

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef matrix<float, 2, 2> FloatMatrix;
typedef matrix<int, 2, 2> IntMatrix;
typedef matrix<uint, 2, 2> UIntMatrix;
typedef vector<float, 2> FloatVector;

float calcResult(FloatMatrix v) 
{ 
    // Multiply diffent parts by different amounts to make order important
    return v[0][0] + v[0][1] * 2 + v[1][0] * 3 + v[1][1] * 4;
}

FloatMatrix makeFloatMatrix(float f)
{
    FloatMatrix m = { { f, f }, { f, f } };
    return m;
}

IntMatrix makeIntMatrix(int v)
{
    IntMatrix m = { { v, v }, { v, v } };
    return m;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);

    float scalarF = idx * (1.0f / (4.0f));

    FloatMatrix ft = {}; 

    FloatMatrix f = { { scalarF + 0.01, scalarF + 0.02f}, { scalarF + 0.011f, scalarF + 0.022f}};
    
    ft += transpose(f);
    
    // fmod
    ft += FloatMatrix(IntMatrix(((f % makeFloatMatrix(0.11f)) * makeFloatMatrix(100)) + makeFloatMatrix(0.5)));
    
    ft += sin(f);
      
    // Lets try some matrix/matrix
    ft = f * ft;
    
    // Lets try some vector matrix
    
    {
        FloatMatrix r = {mul(f[0], ft), mul(ft, f[1])};
        ft += r;
    }
    
    // Back to the transcendentals
   
    ft += cos(f);
    ft += tan(f);
    
    ft += asin(f);
    ft += acos(f);
    ft += atan(f);
    
    ft += atan2(f, makeFloatMatrix(2)); 

#if 0
    // TODO(JS):
    // This fails from DXC with a validation error(!)
    {
        FloatMatrix sf, cf;
        sincos(f, sf, cf);
        
        ft += sf;
        ft += cf;
    }
#endif 
    
    ft += rcp(makeFloatMatrix(1.0) + f);
    ft += FloatMatrix(sign(f - makeFloatMatrix(0.5)));
   
    ft += saturate(f * makeFloatMatrix(4) - makeFloatMatrix(2.0));
    
    ft += sqrt(f);
    ft += rsqrt(makeFloatMatrix(1.0f) + f);
    
    ft += exp2(f);
    ft += exp(f);
    ft += exp10(f);
                
    ft += frac(f * makeFloatMatrix(3));
    ft += ceil(f * makeFloatMatrix(5) - makeFloatMatrix(3));
    
    ft += floor(f * makeFloatMatrix(10) - makeFloatMatrix(7));
    ft += trunc(f * makeFloatMatrix(7));
     
    ft += log(f + makeFloatMatrix(10.0));
    ft += log2(f * makeFloatMatrix(3) + makeFloatMatrix(2));

    {
        float scalarVs[] = { 1, 10, 100, 1000 };
        ft += FloatMatrix(IntMatrix(log10(makeFloatMatrix(scalarVs[idx])) + makeFloatMatrix(0.5f)));
    }
       
    ft += abs(f * makeFloatMatrix(4) - makeFloatMatrix(2.0f));
    
    ft += min(makeFloatMatrix(0.5), f);
    ft += max(f, makeFloatMatrix(0.75));

    ft += pow(makeFloatMatrix(0.5), f);

    ft += smoothstep(makeFloatMatrix(0.2), makeFloatMatrix(0.7), f);
    ft += lerp(makeFloatMatrix(-100), makeFloatMatrix(100), f);

    ft += clamp(f, makeFloatMatrix(0.1), makeFloatMatrix(0.3));

    ft += step(f, makeFloatMatrix(0.5));

    IntMatrix vi = asint(makeFloatMatrix(idx)); 
    ft += asfloat(vi);
    
    UIntMatrix vu = asuint(f);
    ft += asfloat(vu);
   
    outputBuffer[idx] = calcResult(ft);
}
