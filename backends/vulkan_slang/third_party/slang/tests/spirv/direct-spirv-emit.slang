// direct-spirv-emit.slang

//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry computeMain -stage compute -emit-spirv-directly

// Test ability to directly output SPIR-V

// CHECK: OpCapability Shader
// CHECK: OpMemoryModel Logical GLSL450
// CHECK: OpEntryPoint GLCompute %computeMain "main"
// CHECK: OpExecutionMode %computeMain LocalSize 4 1 1

[numthreads(4,1,1)]
void computeMain()
{}
