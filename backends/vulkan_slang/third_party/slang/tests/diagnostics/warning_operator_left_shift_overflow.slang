//TEST:SIMPLE(filecheck=CHECK): -stage compute -entry computeMain -target hlsl

RWStructuredBuffer<int> inputBuffer;
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    uint8_t u8 = uint8_t(inputBuffer[0]);

    uint result = (u8 << 0)
        + (u8 << 1)
        + (u8 << 2)
        + (u8 << 3)
        + (u8 << 4)
        + (u8 << 5)
        + (u8 << 6)
        + (u8 << 7)
        // CHECK-NOT: warning 41030:
        // CHECK: ([[#@LINE+1]]): warning 41030: {{.*}}uint8{{.*}}8
        + (u8 << 8)
        // CHECK: ([[#@LINE+1]]): warning 41030: {{.*}}uint8{{.*}}9
        + (u8 << 9)
        ;

    outputBuffer[0] = int(result);
}
