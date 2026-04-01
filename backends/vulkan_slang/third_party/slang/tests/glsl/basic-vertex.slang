//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage vertex -entry main -allow-glsl
#version 310 es
layout(location = 0) in highp vec4 a_position;
layout(location = 1) in highp vec4 a_coords;
layout(location = 0) out mediump vec3 v_color;
layout(std140, set = 0, binding = 0) uniform buff0 {
    mediump int ui_zero;
};
layout(std140, set = 0, binding = 1) uniform buff1 {
    mediump int ui_one;
};
layout(std140, set = 0, binding = 2) uniform buff2 {
    mediump int ui_two;
};
layout(std140, set = 0, binding = 3) uniform buff3 {
    mediump int ui_three;
};
layout(std140, set = 0, binding = 4) uniform buff4 {
    mediump int ui_four;
};
layout(std140, set = 0, binding = 5) uniform buff5 {
    mediump int ui_five;
};
layout(std140, set = 0, binding = 6) uniform buff6 {
    mediump int ui_six;
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
    // CHECK: OpEntryPoint
}

