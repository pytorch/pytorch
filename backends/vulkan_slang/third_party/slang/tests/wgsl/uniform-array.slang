//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -wgpu -output-using-type

struct MyPoint { int x; int y; int type; }

struct Params{
    MyPoint data[2];
}

//TEST_INPUT:set params = cbuffer(data=[0 1 2 3 4 5 6 7 8 9 10 11])
ConstantBuffer<Params> params;

//TEST_INPUT:set outputBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<float> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    // CHECK: 5
    outputBuffer[0] = params.data[1].y;
}

