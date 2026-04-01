//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target spirv -stage vertex -entry main -allow-glsl
#version 310 es
layout(location = 0) in highp vec4 a_position;
layout(location = 1) in highp vec4 a_coords;
layout(location = 0) out mediump vec3 v_color;


// CHECK: error 31216
layout(std140, not_a_set = 0, binding = 0) uniform buff0 {
    mediump int ui_zero;
};

void main()
{
    gl_Position = a_position;
    mediump vec4 coords = a_coords;
    mediump vec4 res = coords;
    mediump int i = 0;
    for (;;) { res = res.yzwx + vec4(1.0); if (i == 1) break; i++; }
    res -= vec4(2);
    v_color = res.rgb;
}

