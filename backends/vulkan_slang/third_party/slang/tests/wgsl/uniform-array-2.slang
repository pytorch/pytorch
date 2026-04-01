//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -wgpu -output-using-type

struct Params{
    int2 data[2];
    int scalarData[2];
}

//TEST_INPUT:set params = cbuffer(data=[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15])
ConstantBuffer<Params> params;

//TEST_INPUT:set outputBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<float> outputBuffer;

float helper(float2 v)
{return v.y;}

[numthreads(1,1,1)]
void computeMain()
{
    // CHECK: 17
    outputBuffer[0] = helper(params.data[1]) + params.scalarData[1];
}

