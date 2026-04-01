//TEST:SIMPLE(filecheck=GLSL): -target glsl -entry vertMain -stage vertex
//TEST:SIMPLE(filecheck=CHECK): -target spirv
//TEST:SIMPLE(filecheck=CHECK): -target spirv  -emit-spirv-via-glsl

// CHECK-DAG: %[[REG1:[0-9a-zA-Z_]+]] = OpLoad %int %gl_BaseInstance
// CHECK-DAG: %[[REG2:[0-9a-zA-Z_]+]] = OpLoad %int %gl_InstanceIndex
// CHECK-DAG: OpISub %int %[[REG2]] %[[REG1]]

// GLSL: gl_InstanceIndex - gl_BaseInstance

[shader("vertex")]
float4 vertMain(int i : SV_InstanceID) : SV_Position
{
    return i;
}
