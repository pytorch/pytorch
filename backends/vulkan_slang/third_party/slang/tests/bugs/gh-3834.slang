//TEST:SIMPLE(filecheck=CHECK): -stage compute -entry computeMain -target glsl -allow-glsl
//TEST:SIMPLE(filecheck=CHECK): -stage compute -entry computeMain -target glsl -allow-glsl -DUSE_GENERIC

// This tests an issue that "const" variable was losing the
// const-ness when they are in a generic function

uniform sampler2D uniform_sampler2D;
RWStructuredBuffer<float> outputBuffer;

#ifdef USE_GENERIC
__generic<T:__BuiltinArithmeticType>
#endif
vec4 MyFunc()
{
    //CHECK: const ivec2 offset0
    //CHECK: const ivec2 offset1
    const ivec2 offset0 = ivec2(0,0);
    const ivec2 offset1 = ivec2(1,1);
    const ivec2[4] offsets = { offset0, offset0, offset1, offset1 };
    return textureGatherOffsets(uniform_sampler2D, vec2(0), offsets);
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
#ifdef USE_GENERIC
    vec4 result = MyFunc<float>();
#else
    vec4 result = MyFunc();
#endif

    outputBuffer[0] = result[0];
}
