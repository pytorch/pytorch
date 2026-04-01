// We currently don't support matrix intrinsics on glsl based targets
// There are problems with double on dx12/dxbc so we disable that

//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -output-using-type -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile sm_6_0 -use-dxil -output-using-type -shaderobj -render-feature hardware-device
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -output-using-type -shaderobj -render-feature double
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<double> outputBuffer;

typedef double Float;

typedef matrix<Float, 2, 2> FloatMatrix;
typedef matrix<int64_t, 2, 2> IntMatrix;
typedef matrix<uint64_t, 2, 2> UIntMatrix;
typedef vector<Float, 2> FloatVector;

Float calcTotal(FloatVector v)
{
    return v.x + v.y;
}

Float calcTotal(FloatMatrix v) 
{ 
    return calcTotal(v[0]) + calcTotal(v[1]);
}

FloatMatrix makeFloatMatrix(Float f)
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

    Float scalarF = idx * (1.0f / (4.0f));

    FloatMatrix ft = {}; 

    FloatMatrix f = { { scalarF + 0.01, scalarF + 0.02}, { scalarF + 0.011, scalarF + 0.022}};
    
    // fmod - doesn't work on double on D3D/DX
    // ft += FloatMatrix(IntMatrix(((f % makeFloatMatrix(0.11f)) * makeFloatMatrix(100)) + makeFloatMatrix(0.5)));
    
    ft += rcp(makeFloatMatrix(1.0l) + f);
    ft += FloatMatrix(sign(f - makeFloatMatrix(0.5l)));
    
    // Lets try some matrix/matrix
    ft = f * ft;
    
    // Lets try some vector matrix
    
    {
        FloatMatrix r = {mul(f[0], ft), mul(ft, f[1])};
        ft += r;
    }
    
    ft += saturate(f * makeFloatMatrix(4) - makeFloatMatrix(2.0l));
       
    ft += abs(f * makeFloatMatrix(4) - makeFloatMatrix(2.0l));
           
    ft += min(makeFloatMatrix(0.5l), f);
    ft += max(f, makeFloatMatrix(0.75l));

    ft += clamp(f, makeFloatMatrix(0.1l), makeFloatMatrix(0.3l));
    
    outputBuffer[idx] = calcTotal(ft);
}