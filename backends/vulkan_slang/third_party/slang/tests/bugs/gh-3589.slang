//TEST:SIMPLE(filecheck=CHECK): -target hlsl -entry main -profile cs_6_0
RWStructuredBuffer<int64_t> outputBuffer;
RWStructuredBuffer<double> outputBuffer2;

void main(uint id: SV_DispatchThreadID)
{
    let i = int64_t.maxValue;
    let m = int64_t.minValue;
    // CHECK: 9223372036854775807LL
    // CHECK: -9223372036854775808LL
    outputBuffer[0] = i;
    outputBuffer[1] = m;

    // CHECK: -1.79769313486231571e+308
    outputBuffer2[0] = double.minValue;
    // CHECK: 1.79769313486231571e+308
    outputBuffer2[1] = double.maxValue;
}