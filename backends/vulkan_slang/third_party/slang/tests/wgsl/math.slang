//TEST:SIMPLE(filecheck=WGSL): -stage compute -entry computeMain -target wgsl

RWStructuredBuffer<int> inputBuffer;
RWStructuredBuffer<int> outputBuffer;

__generic<T:__BuiltinFloatingPointType>
bool Test_Scalar()
{
    // WGSL-LABEL: Test_Scalar

    const T zero = T(inputBuffer[0]);
    const T one = T(inputBuffer[1]);
    const int zeroInt = int(inputBuffer[0]);

    T outFloat1, outFloat2;
    int outInt;

    return true

        // WGSL: acos(
        && zero == acos<T>(one)

        // WGSL: acosh(
        && zero == acosh<T>(one)

        // WGSL: asin(
        && zero == asin<T>(zero)

        // WGSL: asinh(
        && zero == asinh<T>(zero)

        // WGSL: atan(
        && zero == atan<T>(zero)

        // WGSL: atan2(
        && zero == atan2<T>(zero, zero)

        // WGSL: atanh(
        && zero == atanh<T>(zero)

        // WGSL: ceil(
        && zero == ceil<T>(zero)

        // WGSL: cos(
        && one == cos<T>(zero)

        // WGSL: cosh(
        && one == cosh<T>(zero)

        // WGSL: exp(
        && one == exp<T>(zero)

        // WGSL: exp2(
        && one == exp2<T>(zero)

        // WGSL: abs(
        && zero == abs<T>(zero)

        // WGSL: floor(
        && zero == floor<T>(zero)

        // WGSL: fma(
        && zero == fma(zero, zero, zero)

        // WGSL: max(
        && zero == max<T>(zero, zero)

        // WGSL: min(
        && zero == min<T>(zero, zero)

        // WGSL: fract(
        && zero == fract<T>(zero)

        // WGSL: frexp(
        && zero == frexp<T>(zero, outInt) && zeroInt == outInt

        // WGSL: ldexp(
        && zero == ldexp<T>(zero, zeroInt)

        // WGSL: log(
        && zero == log<T>(one)

        // WGSL: log2(
        && zero == log2<T>(one)

        // WGSL: modf(
        && zero == modf<T>(zero, outFloat1)

        // WGSL: pow(
        && zero == pow<T>(zero, one)

        // WGSL: round(
        && zero == round<T>(zero)

        // WGSL: sin(
        && zero == sin<T>(zero)

        // WGSL: sinh(
        && zero == sinh<T>(zero)

        // WGSL: sqrt(
        && zero == sqrt<T>(zero)

        // WGSL: tan(
        && zero == tan<T>(zero)

        // WGSL: tanh(
        && zero == tanh<T>(zero)

        // WGSL: trunc(
        && zero == trunc<T>(zero)
        ;
}

__generic<T:__BuiltinFloatingPointType, let N : int>
bool Test_Vector()
{
    // WGSL-LABEL: Test_Vector_0
    const vector<T,N> zero = T(inputBuffer[0]);
    const vector<T,N> one = T(inputBuffer[1]);

    const vector<int,N> zeroInt = int(inputBuffer[0]);

    vector<T,N> outFloat1, outFloat2;
    vector<int,N> outInt;

    return true
        // WGSL: acos(
        // WGSL-NOT: acos(
        && zero == acos<T>(one)

        // WGSL: acosh(
        // WGSL-NOT: acosh(
        && zero == acosh<T>(one)

        // WGSL: asin(
        // WGSL-NOT: asin(
        && zero == asin<T>(zero)

        // WGSL: asinh(
        // WGSL-NOT: asinh(
        && zero == asinh<T>(zero)

        // WGSL: atan(
        // WGSL-NOT: atan(
        && zero == atan<T>(zero)

        // WGSL: atan2(
        // WGSL-NOT: atan2(
        && zero == atan2<T>(zero, zero)

        // WGSL: atanh(
        // WGSL-NOT: atanh(
        && zero == atanh<T>(zero)

        // WGSL: ceil(
        // WGSL-NOT: ceil(
        && zero == ceil<T>(zero)

        // WGSL: cos(
        // WGSL-NOT: cos(
        && one == cos<T>(zero)

        // WGSL: cosh(
        // WGSL-NOT: cosh(
        && one == cosh<T>(zero)

        // WGSL: exp(
        // WGSL-NOT: exp(
        && one == exp<T>(zero)

        // WGSL: exp2(
        // WGSL-NOT: exp2(
        && one == exp2<T>(zero)

        // WGSL: abs(
        // WGSL-NOT: abs(
        && zero == abs<T>(zero)

        // WGSL: floor(
        // WGSL-NOT: floor(
        && zero == floor<T>(zero)

        // WGSL: fma(
        // WGSL-NOT: fma(
        && zero == fma(zero, zero, zero)

        // WGSL: max(
        // WGSL-NOT: max(
        && zero == max<T>(zero, zero)

        // WGSL: min(
        // WGSL-NOT: min(
        && zero == min<T>(zero, zero)

        // WGSL: fract(
        // WGSL-NOT: fract(
        && zero == fract<T>(zero)

        // WGSL: frexp(
        // WGSL-NOT: frexp(
        && zero == frexp<T>(zero, outInt) && all(zeroInt == outInt)

        // WGSL: ldexp(
        // WGSL-NOT: ldexp(
        && zero == ldexp<T>(zero, zeroInt)

        // WGSL: log(
        // WGSL-NOT: log(
        && zero == log<T>(one)

        // WGSL: log2(
        // WGSL-NOT: log2(
        && zero == log2<T>(one)

        // WGSL: modf(
        // WGSL-NOT: modf(
        && zero == modf<T>(zero, outFloat1)

        // WGSL: pow(
        // WGSL-NOT: pow(
        && zero == pow<T>(zero, one)

        // WGSL: round(
        // WGSL-NOT: round(
        && zero == round<T>(zero)

        // WGSL: sin(
        // WGSL-NOT: sin(
        && zero == sin<T>(zero)

        // WGSL: sinh(
        // WGSL-NOT: sinh(
        && zero == sinh<T>(zero)

        // WGSL: sqrt(
        // WGSL-NOT: sqrt(
        && zero == sqrt<T>(zero)

        // WGSL: tan(
        // WGSL-NOT: tan(
        && zero == tan<T>(zero)

        // WGSL: tanh(
        // WGSL-NOT: tanh(
        && zero == tanh<T>(zero)

        // WGSL: trunc(
        // WGSL-NOT: trunc(
        && zero == trunc<T>(zero)
        ;

    // WGSL-LABEL: Test_Vector_1
}

[numthreads(1,1,1)]
void computeMain()
{
    // GLSL: void main(
    // GLSL_SPIRV: OpEntryPoint
    // SPIR: OpEntryPoint
    // HLSL: void computeMain(
    // CUDA: void computeMain(
    // CPP: void _computeMain(

    bool result = true
        && Test_Scalar<float>()
        && Test_Vector<float, 2>()
        && Test_Vector<float, 3>()
        && Test_Vector<float, 4>()
        && Test_Scalar<half>()
        && Test_Vector<half, 2>()
        && Test_Vector<half, 3>()
        && Test_Vector<half, 4>()
        ;

    // BUF: 1
    outputBuffer[0] = int(result);
}
