//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage vertex -entry main -allow-glsl
#version 310 es
layout(location = 0) in highp vec4 a_position;

void main()
{
    vec4 a = 0;
    for (highp int i = 0; i < 5; i++)
        a += vec4(i);
    gl_Position = a;
   
    // CHECK: OpEntryPoint
}

