//TEST:SIMPLE(filecheck=SPIRV): -target spirv -stage vertex -entry main -allow-glsl -emit-spirv-directly
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -stage vertex -entry main -allow-glsl
//TEST:SIMPLE(filecheck=METAL): -target metal -stage vertex -entry main -allow-glsl -matrix-layout-row-major
#version 310 es
layout(location = 0) in highp vec4 a_position;

layout(std140, set = 0, binding = 0) uniform buff0 {
    mat4x4 m1;
    mat4x4 m2;
};


void main()
{
    gl_Position = m1 * m2 * a_position;
   
    // SPIRV: OpEntryPoint
    // SPIRV: OpMatrixTimesMatrix
    // SPIRV: OpMatrixTimesVector

    // METAL: {{.*}}m1{{.*}}*{{.*}}m2{{.*}}*{{.*}}a_position{{.*}}
}

