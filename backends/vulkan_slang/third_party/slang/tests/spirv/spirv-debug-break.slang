// spirv-instruction.slang
//TEST(compute, vulkan):SIMPLE(filecheck=CHECK):-target glsl -entry computeMain -stage compute

[[vk::spirv_instruction(1, "NonSemantic.DebugBreak")]]
void _spvDebugBreak(int v);

[ForceInline] 
void _debugBreak() { _spvDebugBreak(1); }

// CHECK:spirv_instruction(id = 1, set = "NonSemantic.DebugBreak")
// CHECK:void _spvDebugBreak_0(int _0);
// CHECK:void main()
// CHECK:_spvDebugBreak_0(1);

//TEST_INPUT:set resultBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<uint> resultBuffer;

[numthreads(4,1,1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint threadId = dispatchThreadID.x;
    
    _debugBreak();
    
    resultBuffer[threadId] = threadId + threadId;
}
