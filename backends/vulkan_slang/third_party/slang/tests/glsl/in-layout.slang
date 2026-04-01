//TEST:SIMPLE(filecheck=CHECK): -target spirv -entry main -stage fragment

// Test that we can mix explicit and implicit locations for global varyings.

#version 450
layout(location=1) in vec4 color_0;
in vec2 texcoord_0;
in vec2 texCoord_1;

out vec4 out1;

// CHECK-DAG: OpDecorate %color_0 Location 1

// CHECK-DAG: OpDecorate %texcoord_0 Location 0

// CHECK-DAG: OpDecorate %texCoord_1 Location 2

// CHECK-DAG: OpDecorate %entryPointParam_main_out1 Location 0

void main()
{
    out1 = vec4(color_0.rg, texcoord_0 + texCoord_1);
}