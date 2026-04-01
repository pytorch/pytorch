//TEST:SIMPLE(filecheck=SPV): -allow-glsl -target spirv-asm -entry vertexMain -stage vertex

// This test is to make sure the constant-folding works for the switch-case label.
// The shader code is from VK-CTS but modified,
//   dEQP-VK.glsl.switch.const_expr_in_label_dynamic_fragment

layout(location = 0) in highp vec4 a_position;
layout(location = 1) in highp vec4 a_coords;

layout(location = 0) out mediump vec4 v_color;
layout (std140, set=0, binding=0) uniform buffer0 { highp int ui_two; };

void vertexMain(void)
{
    gl_Position = a_position;
    highp vec4 coords = a_coords;
    mediump vec3 res = vec3(0.0);

    const int t = 2;
    switch (ui_two)
    {
        //SPV-NOT:([[# @LINE+1]]): error
        case int(0.0):
            res = coords.xyz;
            break;

        //SPV-NOT:([[# @LINE+1]]): error
        case 2-1:
            res = coords.wzy;
            break;

        //SPV-NOT:([[# @LINE+1]]): error
        case 3&(1<<1):
            res = coords.yzw;
            break;

        //SPV-NOT:([[# @LINE+1]]): error
        case t+1:
            res = coords.zyx;
            break;
    }

    v_color = vec4(res, 1.0);
}

