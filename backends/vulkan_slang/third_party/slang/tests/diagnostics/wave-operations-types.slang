//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -entry computeMain -stage compute -target metal

RWStructuredBuffer<uint> out;

[shader("compute")]
void computeMain(uint3 dispatchID : SV_DispatchThreadID)
{
    // CHECK: Unsupported type for subgroup operations in Metal. Valid types include
    // CHECK: Unsupported type for subgroup operations in Metal. Valid types include
    // CHECK: Unsupported type for subgroup operations in Metal. Valid types include
    out[0] = WaveRotate(true, 1);
    out[1] = WaveRotate(uint8_t(dispatchID.x), 1);
    out[2] = WaveRotate(uint64_t(dispatchID.x), 1);
}
