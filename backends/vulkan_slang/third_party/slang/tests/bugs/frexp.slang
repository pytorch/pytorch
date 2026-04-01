//TEST(smoke,compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-output-using-type
//TEST(smoke,compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-dx12 -use-dxil -output-using-type
//TEST(smoke,compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-cpu -output-using-type
//TEST(smoke,compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -output-using-type
//TEST(smoke,compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -emit-spirv-directly -output-using-type
//TEST(smoke,compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-cuda -output-using-type
//TEST(smoke,compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-slang -shaderobj -mtl -output-using-type

// BUF:      type: int32_t
// BUF-NEXT: 1
// BUF-NEXT: 6
// BUF-NEXT: -1
// BUF-NEXT: 0

//TEST_INPUT:set inputBuffer = ubuffer(data=[1.0 42.75 0.25 0.0], stride=4)
RWStructuredBuffer<float> inputBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint i : SV_GroupIndex)
{
    var exponent : int = 0;
    frexp(inputBuffer[i], exponent);
    outputBuffer[i] = exponent;
}
