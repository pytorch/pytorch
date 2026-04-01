//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target metal
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target wgsl

RWStructuredBuffer<float> outputBuffer;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain()
{
    // CHECK: error 56102: division by matrix is not supported
    float3x3 divisor = float3x3(2.5);
    divisor[1][1] = 1.5;
    outputBuffer[0] = (float3x3(15) / divisor)[0][0];
}

