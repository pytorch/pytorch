//TEST:SIMPLE(filecheck=HLSL): -target hlsl -stage compute -entry computeMain
//TEST:SIMPLE(filecheck=GLSL): -target glsl -stage compute -entry computeMain

// Metal backend requires a new legalization pass to support emitting `TextureBufferType`
//DISABLE_TEST:SIMPLE(filecheck=METAL): -target metal -stage compute -entry computeMain
//DISABLE_TEST:SIMPLE(filecheck=METALLIB): -target metallib -stage compute -entry computeMain

// SPIRV backend has no support for emitting `TextureBufferType`
//DISABLE_TEST:SIMPLE(filecheck=SPIRV): -target spirv -stage compute -entry computeMain

tbuffer tbuf : register(t0)
{
    float4 tb_val1;
}

tbuffer tbuf2 : register(t1)
{
    Texture2D<float4> texture2D;
    float4 tb_val2;
}


// HLSL-DAG: t0
// HLSL-DAG: t1
// HLSL-DAG: t2

// GLSL-DAG: binding = 0
// GLSL-DAG: binding = 1
// GLSL-DAG: binding = 2

// METAL-DAG: [texture(0)]
// METAL-DAG: [texture(1)]
// METAL-DAG: [texture(2)]
// METALLIB: @computeMain

RWStructuredBuffer<float4> outputBuffer;
[numthreads(1,1,1)]
void computeMain() {
    outputBuffer[0] = tb_val1 + texture2D[0] + tb_val2;
}
