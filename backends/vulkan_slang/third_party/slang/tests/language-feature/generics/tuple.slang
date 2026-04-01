//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -wgpu -shaderobj -output-using-type -render-features half

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

T writeSingle<T : __BuiltinFloatingPointType>(inout int offset, T value)
{
    outputBuffer[offset++] = __realCast<float>(value);
    return value;
}

void write<each T : __BuiltinFloatingPointType>(Tuple<T> tuple)
{
    int offset = 0;
    expand writeSingle(offset, each tuple);
}

[numthreads(1,1,1)]
void computeMain()
{
    // CHECK: 1.0
    // CHECK: 2.0
    // CHECK: 3.0
    // CHECK: 4.0
    let tuple = Tuple<float, half, float, float>(1.0, 2.0h, 3.0, 4.0);
    write(tuple);
}
