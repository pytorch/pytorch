//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-dx12 -use-dxil -compute -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -output-using-type

//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly

// To check that our counter-initialization works correctly, set the initial
// counter to 1 instead of 0
//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4, counter=1):out,name=outputBuffer
AppendStructuredBuffer<int> outputBuffer;

//TEST_INPUT:set inBuffer = ubuffer(data=[1 2 3 4], stride=4)
RWStructuredBuffer<int> inBuffer;

// Make sure the bindings are correct. outputBuffer should take two slots, and inBuffer
// should be at binding 2.
// CHECK: OpDecorate %inBuffer Binding 2

[numthreads(4, 1, 1)]
void computeMain(uint i : SV_GroupIndex)
{
    int g = inBuffer[i];
    outputBuffer.Append(g);

    GroupMemoryBarrier();

    uint numStructs, stride;
    outputBuffer.GetDimensions(numStructs, stride);
    if(i == 0)
        outputBuffer.Append(int(numStructs));

    // BUF: type: int32_t
    // Never assigned, as we set the initial counter to 1
    // BUF: 0

    // The values from inBuffer in any order
    // BUF-DAG: 1
    // BUF-DAG: 3
    // BUF-DAG: 2
    // BUF-DAG: 4

    // The total size of the AppendStructuredBuffer (from GetDimensions)
    // BUF: 8

    // Never assigned
    // BUF: 0
    // BUF: 0
}
