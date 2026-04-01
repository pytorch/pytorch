//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK): -output-using-type

static const uint X_SIZE = 32;

uint y<int x> (int y){return 0;}

uint test(uint x)
{
    uint y = 0;
    // With two stage parsing, we should be able to disambiguate this
    // from a generic function call.
    uint surround_mask = x +
        y<3?1:x>(X_SIZE) // generic call or comparison?
        ? 5 : 0;
    return surround_mask;
}

//TEST_INPUT: set outputBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<uint> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    // CHECK: 1
    outputBuffer[0] = test(1);
    // CHECK: 5
    outputBuffer[1] = test(33);
}