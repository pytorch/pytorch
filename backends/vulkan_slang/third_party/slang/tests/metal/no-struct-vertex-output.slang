//TEST:SIMPLE(filecheck=METAL): -target metallib -stage vertex -entry vertexMain
//TEST:SIMPLE(filecheck=METALLIB): -target metallib -stage vertex -entry vertexMain
//TEST:SIMPLE(filecheck=WGSL): -target wgsl -stage vertex -entry vertexMain
//TEST:SIMPLE(filecheck=WGSLSPIRV): -target wgsl-spirv-asm -stage vertex -entry vertexMain

//METAL-DAG: position
//METALLIB: @vertexMain

//WGSL: @builtin(position)
//WGSLSPIRV: %vertexMain = OpFunction

// Vertex Shader

float4 vertexMain()
{
    return float4(1,1,1,1);
}
