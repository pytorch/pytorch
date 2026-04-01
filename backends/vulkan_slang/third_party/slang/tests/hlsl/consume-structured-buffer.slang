//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-dx12 -use-dxil -compute -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -output-using-type -emit-spirv-directly

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

// To check that our counter-initialization works correctly, set the initial
// counter to 6 instead of 8, so we'll start reading from 6.0 downwards
//TEST_INPUT:set consumeBuffer = ubuffer(data=[1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0], stride=4, counter=6)
ConsumeStructuredBuffer<float> consumeBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint i : SV_GroupIndex)
{
    uint size, stride;
    consumeBuffer.GetDimensions(size, stride);

    if(i == 0)
        outputBuffer[0] = size;

    outputBuffer[i+1] = consumeBuffer.Consume();

    // BUF: type: float
    // The total size of the ConsumeStructuredBuffer (from GetDimensions)
    // BUF: 8.0

    // The values from consumeBuffer in any order
    // BUF-DAG: 6.0
    // BUF-DAG: 4.0
    // BUF-DAG: 5.0
    // BUF-DAG: 3.0
}
