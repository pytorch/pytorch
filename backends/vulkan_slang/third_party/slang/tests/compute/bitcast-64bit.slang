//TEST(compute):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -emit-spirv-via-glsl -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -d3d12 -profile cs_6_6 -use-dxil -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<uint64_t> gOutputBuffer;

int64_t icast(double x)
{
    return bit_cast<int64_t>(x);
}

int64_t icast(uint64_t x)
{
    return bit_cast<int64_t>(x);
}

uint64_t ucast(double x)
{
    return bit_cast<uint64_t>(x);
}

uint64_t ucast(int64_t x)
{
    return bit_cast<uint64_t>(x);
}

[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain()
{
    double t1 = -1.0;
    uint64_t t2 = 2;
    gOutputBuffer[0] = icast(t1);    // 0xBFF0000000000000 => 13830554455654793216
    gOutputBuffer[1] = icast(t2);    // 0x0000000000000002 => 2

    double t3 = 3.0;
    int64_t t4 = -4;
    gOutputBuffer[2] = ucast(t3);   // 0x4008000000000000 => 4613937818241073152
    gOutputBuffer[3] = ucast(t4);   // 0xFFFFFFFFFFFFFFFC => 18446744073709551612
}
