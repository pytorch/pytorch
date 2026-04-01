//TEST:SIMPLE(filecheck=CHECK): -target spirv

// Test that we can use intrinsics in global scope constant array, which causes
// the spirv_asm to be inlined in global module scope.
// Our global value inlining pass should be able to clean up the global scope spirv_asm
// blocks and inlining them to use sites.

// CHECK: %main = OpFunction
// CHECK: OpStore

static const uint staticArr[] = {
    uint((((uint)round(saturate(1) * 255) << 24) | ((uint)round(saturate(0) * 255) << 16) | ((uint)round(saturate(0) * 255) << 8) | 0xff)),
    uint((((uint)round(saturate(1) * 255) << 24) | ((uint)round(saturate(0) * 255) << 16) | ((uint)round(saturate(1) * 255) << 8) | 0xff))
};

RWStructuredBuffer<int> buffer;

[numthreads(1,1,1)]
void main(int i : SV_DispatchThreadID)
{
    buffer[0] = staticArr[i] + staticArr[1];
}
