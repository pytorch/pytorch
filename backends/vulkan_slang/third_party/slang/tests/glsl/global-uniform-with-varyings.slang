//TEST:SIMPLE(filecheck=CHECK_SPIRV): -target spirv -entry main -stage vertex
//TEST:SIMPLE(filecheck=CHECK_GLSL): -target glsl -entry main -stage vertex

// CHECK_SPIRV: OpEntryPoint
// CHECK_SPIRV: OpVariable {{.*}} Input {{.*}} Location 0
// CHECK_SPIRV: OpVariable {{.*}} Uniform
// CHECK_SPIRV: OpVariable {{.*}} Input {{.*}} Location 1
// CHECK_SPIRV: OpVariable {{.*}} Output {{.*}} Location 0

// CHECK_GLSL: layout(location = 0)
// CHECK_GLSL-NEXT: in
// CHECK_GLSL: layout(location = 1)
// CHECK_GLSL-NEXT: in
// CHECK_GLSL: layout(std140) uniform
// CHECK_GLSL: layout(location = 0)
// CHECK_GLSL-NEXT: out
// CHECK_GLSL: void main

#version 330 core
layout(location=0) in float3 a_position;
layout(location=1) in float4 a_color;

out float4 v_color;

uniform matrix<float,4,4> u_transform;

void main() {
    gl_Position = u_transform * vec4(a_position, 1.0);
    v_color = a_color;
}
