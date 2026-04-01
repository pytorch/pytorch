//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -cpu -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct S<T>
{
    T v;
    __init(T x)
    {
        v = x;
    }
}


S<T> makeS<T>(T x)
{
    return S<T>(x);
}

void accept<each T>(expand each T value) {}

void writeSingle<T : __BuiltinFloatingPointType>(T value)
{
    static int counter = 0;
    outputBuffer[counter++] = __realCast<float>(value);
    //return value;
}

void write<each T : __BuiltinFloatingPointType>(expand S<each T> value)
{
    accept(expand writeSingle((each value).v));
}

[numthreads(1,1,1)]
void computeMain()
{
    // CHECK: 1.0
    // CHECK: 2.0
    // CHECK: 3.0
    // CHECK: 4.0
    write(makeS(1.0), makeS(2.0), makeS(3.0), makeS(4.0));
    write();
}