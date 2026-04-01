//TEST:SIMPLE(filecheck=CHECK): -entry missShader -entry missShader2 -target spirv
//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK: OpEntryPoint MissKHR %missShader "missShader"
// CHECK: OpEntryPoint MissKHR %missShader2 "missShader2"

struct RayPayload{};
[shader("miss")]
void missShader(inout RayPayload payload) { }

[shader("miss")]
void missShader2(inout RayPayload payload)  { }