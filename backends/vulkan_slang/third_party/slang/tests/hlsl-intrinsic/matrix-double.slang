// TODO(JS):
// It doesn't look like fxc, dxc, vk support double versions of many of the intrinsics, so they are disabled here.
// Arguably we should implement simple intrinsics if missing in the core module.
// More complicated functions (like say sin) can also be written, if not available on a target, but requires significant
// care.

// TODO(JS):
// NOTE! war-double-host-callable category is a work around for issues with x86 & host-callable

//TEST(compute, war-double-host-callable):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -output-using-type
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -output-using-type
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -output-using-type -render-feature double
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -output-using-type

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
    return FloatMatrix(f);
}

IntMatrix makeIntMatrix(int v)
{
    IntMatrix m = { { v, v }, { v, v } };
    return m;
}

void test1(inout FloatMatrix ft, inout FloatMatrix f, int idx)
{
    unmodified(f);

    // fmod
    ft += FloatMatrix(IntMatrix(((f % makeFloatMatrix(0.11f)) * makeFloatMatrix(100)) + makeFloatMatrix(0.5)));

    ft += sin(f);

    // Lets try some matrix/matrix
    ft = f * ft;

    // Lets try some vector matrix

    {
        FloatMatrix r = { mul(f[0], ft), mul(ft, f[1]) };
        ft += r;
    }

    // Back to the transcendentals

    ft += cos(f);
    ft += tan(f);

    ft += asin(f);
    ft += acos(f);
    ft += atan(f);

    ft += atan2(f, makeFloatMatrix(2));
    {
        FloatMatrix sf, cf;
        sincos(f, sf, cf);

        ft += sf;
        ft += cf;
    }

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
        ft += FloatMatrix(IntMatrix(log10(makeFloatMatrix(Float(scalarVs[idx]))) + makeFloatMatrix(0.5f)));
    }

    ft += abs(f * makeFloatMatrix(4) - makeFloatMatrix(2.0f));

    ft += min(makeFloatMatrix(0.5), f);
    ft += max(f, makeFloatMatrix(0.75));
}

void test2(inout FloatMatrix ft, inout FloatMatrix f)
{
    unmodified(f);

    ft += pow(makeFloatMatrix(0.5), f);

    ft += smoothstep(makeFloatMatrix(0.2), makeFloatMatrix(0.7), f);
    ft += lerp(makeFloatMatrix(-100), makeFloatMatrix(100), f);

    ft += clamp(f, makeFloatMatrix(0.1), makeFloatMatrix(0.3));

    ft += step(f, makeFloatMatrix(0.5));

#if 0
    IntMatrix vi = asint(makeFloatMatrix(idx)); 
    ft += asfloat(vi);
    
    UIntMatrix vu = asuint(f);
    ft += asfloat(vu);
#endif
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);

    Float scalarF = idx * (1.0f / (4.0f));

    FloatMatrix ft = {}; 

    FloatMatrix f = { { scalarF + 0.01, scalarF + 0.02}, { scalarF + 0.011, scalarF + 0.022}};

    test1(ft, f, idx);
    test2(ft, f);
    
    outputBuffer[idx] = calcTotal(ft);
}
