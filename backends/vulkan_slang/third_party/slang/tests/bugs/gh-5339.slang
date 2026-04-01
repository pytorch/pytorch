//TEST:SIMPLE(filecheck=SPV): -allow-glsl -target spirv-asm -entry computeMain -stage compute

// Test if we are correctly using `Offset` option instead of `ConstOffset`
// when the offset value is not a compile-time constant.

//SPV:OpCapability ImageGatherExtended

#extension GL_EXT_gpu_shader5 : require

layout (location = 0) in highp vec2 v_texCoord;

layout (binding = 0) uniform highp sampler2D u_sampler;
layout (binding = 1) uniform offset { highp ivec2 u_offset; };

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
buffer MyBlockName
{
    vec4 result;
} outputBuffer;

void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //SPV:OpImageGather %
    //SPV-NOT:Const
    //SPV-SAME: Offset %
    outputBuffer.result = textureGatherOffset(u_sampler, v_texCoord, u_offset);
}
