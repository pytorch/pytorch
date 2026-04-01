//TEST:SIMPLE(filecheck=CHECK1): -entry main1 -entry main2 -target spirv -fvk-use-entrypoint-name -emit-spirv-directly
//TEST:SIMPLE(filecheck=CHECK2): -target spirv -fvk-use-entrypoint-name -emit-spirv-directly


[shader("raygeneration")]
void main1() {}

[shader("raygeneration")]
void main2() {}

[shader("raygeneration")]
void main3() {}

// CHECK1: OpEntryPoint
// CHECK1: OpEntryPoint
// CHECK1-NOT: OpEntryPoint

// CHECK2: OpEntryPoint
// CHECK2: OpEntryPoint
// CHECK2: OpEntryPoint
