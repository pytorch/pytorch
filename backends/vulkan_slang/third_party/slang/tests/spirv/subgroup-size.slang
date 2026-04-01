import "glsl";

//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -O0

// CHECK-DAG: %[[CONST:[A-Za-z0-9_]+]] = OpConstantComposite %v3int %int_1 %int_2 %int_3
// CHECK: OpBitcast %v3uint %[[CONST]]

RWStructuredBuffer<int> outputBuffer;

[shader("compute")]
[numthreads(1u, 2u, 3)]
void compute()
{
    const int x = gl_WorkGroupSize.x;
    outputBuffer[0] = x;
}