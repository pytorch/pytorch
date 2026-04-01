//TEST:SIMPLE(filecheck=CHECK):-target spirv -emit-spirv-directly

// CHECK: error 55200:

RWStructuredBuffer<float> outputBuffer;

float testFunc(vector<vector<float, 3>, 2> v1)
{
    return v1.x.x;
}

[numthreads(1,1,1)]
void main(uint v : SV_DispatchThreadID)
{
    vector<vector<float, 3>, 2> v1 = { { 1, 2, 3 }, { 4, 5, v } };
    outputBuffer[0] = testFunc(v1);
}