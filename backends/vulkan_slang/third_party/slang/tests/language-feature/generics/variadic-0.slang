//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -cpu -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -wgpu -shaderobj -output-using-type -render-features half

//TEST:SIMPLE(filecheck=CHECK): -target glsl
// CHECK-NOT: NullDifferential

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct S<T:__BuiltinArithmeticType>
{
    T v;
    __init(T x)
    {
        v = x;
    }
}


S<T> makeS<T:__BuiltinArithmeticType>(T x)
{
    return S<T>(x);
}
bool cmp<T:__BuiltinArithmeticType>(T a, int b)
{
    return a > T(b);
}
void accept<each T>(expand each T value) {}

T writeSingle<T : __BuiltinFloatingPointType>(inout int offset, T value)
{
    outputBuffer[offset++] = __realCast<float>(value);
    return value;
}

void write<each T : __BuiltinFloatingPointType>(expand S<each T> value)
{
    int offset = 0;
    accept(expand writeSingle(offset, cmp((each value).v, 0) ? (each value).v : __realCast<each T>(0.0)));
}

[numthreads(1,1,1)]
void computeMain()
{
    // CHECK: 1.0
    // CHECK: 2.0
    // CHECK: 3.0
    // CHECK: 4.0
    write();
    write(makeS(1.0), makeS(2.0h), makeS(3.0), makeS(4.0));
}
