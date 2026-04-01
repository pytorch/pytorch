//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target wgsl -entry computeMain -stage compute

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out, name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

cbuffer VectorBuffer
{
    // CHECK: error 38203
    vector<int8_t, 2> int8Vector;
    // CHECK: error 38203
    vector<int16_t, 2> int16Vector;
    // CHECK: error 38203
    vector<int64_t, 3> int64Vector;
};

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer[0] = int8Vector.x + int16Vector.y + int64Vector.z;
}