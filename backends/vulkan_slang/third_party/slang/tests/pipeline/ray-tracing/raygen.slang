// raygen.slang

//TEST(optix):COMPARE_COMPUTE_EX:-cuda -rt -output-using-type -compute-dispatch 4,1,1

//TEST_INPUT:ubuffer(data=[0 0 0 0],stride=4):name gOutput,out

RWStructuredBuffer<int> gOutput;

[shader("raygeneration")]
void raygenMain()
{
    uint3 tid = DispatchRaysIndex();

    gOutput[tid.x] = tid.x*11;
}
