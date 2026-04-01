//TEST:SIMPLE(filecheck=METAL): -target metal
//TEST:SIMPLE(filecheck=WGSL): -target wgsl -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=GLSL): -target glsl -allow-glsl
//TEST:SIMPLE(filecheck=GLSL): -target glsl
//TEST:SIMPLE(filecheck=CHECK): -target spirv -allow-glsl
//TEST:SIMPLE(filecheck=CHECK): -target spirv

// METAL-DAG: constant bool fc_constValue2{{.*}} {{\[\[}}function_constant(1){{\]\]}};
// METAL-DAG: constant bool constValue2{{.*}} = is_function_constant_defined(fc_constValue2{{.*}}) ? fc_constValue2{{.*}} : true;

// METAL-DAG: constant float fc_constValue1{{.*}} {{\[\[}}function_constant(7){{\]\]}};

// METAL-DAG: constant int fc_constValue0{{.*}} {{\[\[}}function_constant(0){{\]\]}};

// METAL-DAG: constant int fc_constValue3{{.*}} {{\[\[}}function_constant(9){{\]\]}};

// WGSL-DAG: @id(0) override constValue0{{.*}} = {{.*}}

// WGSL-DAG: @id(7) override constValue1{{.*}} = {{.*}}

// WGSL-DAG: @id(1) override constValue2{{.*}} = {{.*}}

// WGSL-DAG: @id(9) override constValue3{{.*}} = {{.*}}

// CHECK-DAG: OpDecorate %[[C0:[0-9A-Za-z_]+]] SpecId 0
// CHECK-DAG: %[[C0]] = OpSpecConstant %int 1

// CHECK-DAG: OpDecorate %[[C1:[0-9A-Za-z_]+]] SpecId 7
// CHECK-DAG: %[[C1]] = OpSpecConstant %float 3

// CHECK-DAG: OpDecorate %[[C2:[0-9A-Za-z_]+]] SpecId 1
// CHECK-DAG: %[[C2]] = OpSpecConstantTrue %bool

// CHECK-DAG: OpDecorate %[[C3:[0-9A-Za-z_]+]] SpecId 9
// CHECK-DAG: %[[C3]] = OpSpecConstant %int 111

// GLSL-DAG: layout(constant_id = 0)
// GLSL-DAG: int constValue0_0 = 1;

// GLSL-DAG: layout(constant_id = 7)
// GLSL-DAG: float constValue1_0 = 3.0;

// GLSL-DAG: layout(constant_id = 1)
// GLSL-DAG: bool constValue2_0 = true;

// GLSL-DAG: layout(constant_id = 9)
// GLSL-DAG: int constValue3_0 = 111;

[vk::specialization_constant]
const int constValue0 = 1;

[vk::constant_id(7)]
const float constValue1 = 3.0f;

[SpecializationConstant]
const bool constValue2 = true;

layout(constant_id = 9) const int constValue3 = 111;

RWStructuredBuffer<float> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain()
{
    if (constValue2)
        outputBuffer[0] = constValue0 + (int)constValue1;
    else
        outputBuffer[0] = constValue3;
}