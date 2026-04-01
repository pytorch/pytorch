//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage fragment -entry fragmentMain -allow-glsl
//TEST:SIMPLE(filecheck=CHECK_GLSL): -target glsl -stage fragment -entry fragmentMain -allow-glsl

#version 450

static const uint BINDING_BASE = 3;
static const uint BINDING_STRIDE = 2;
static const uint SET_BASE = 1;
static const uint SET_STRIDE = 4;
static const uint BASE_IN_LOC = 3;
static const uint BASE_IN_INDEX = 24;
static const uint BASE_OUT_LOC = 2;
static const uint INPUT_ATT_BASE = 16;

// CHECK_GLSL: binding = 18
// CHECK_GLSL: set = 9
layout(set = SET_BASE + SET_STRIDE * 2, binding = BINDING_BASE * BINDING_STRIDE * 3) buffer Output {
    float result[];
};

layout(set = SET_BASE, binding = BINDING_BASE * BINDING_STRIDE) buffer InputA {
    float a[];
};

layout(set = SET_BASE + SET_STRIDE, binding = BINDING_BASE * BINDING_STRIDE * 2) buffer InputB {
    float b[];
};

// CHECK_GLSL: binding = 3
// CHECK_GLSL: set = 6
// CHECK_GLSL: input_attachment_index = 10
layout(input_attachment_index = (INPUT_ATT_BASE + 4) / 2, set = SET_BASE + 5, binding = BINDING_BASE) uniform subpassInput inputAttachment0;

// CHECK_GLSL: constant_id = 27
layout(constant_id = 3 * 9) const float specConst = 0.2;

// CHECK_GLSL: location = 6
[vk_location(BASE_OUT_LOC + 4)]
out vec4 fragColor;

// CHECK_GLSL: location = 7
layout(location = (BASE_IN_LOC * 2) + 1, index = BASE_IN_INDEX / 3 ) in float inParam;

void fragmentMain() {
    uint index = gl_FragCoord.x;
    result[index] = a[index] * b[index];

    vec4 attachmentData = subpassLoad(inputAttachment0);
    fragColor = vec4(attachmentData.rgb, 1.0) * specConst * inParam;

    // CHECK: OpEntryPoint
    // CHECK_GLSL: main
}

