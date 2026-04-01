//TEST:SIMPLE(filecheck=CHECK_GLSL): -stage compute -entry computeMain -allow-glsl -target glsl
//TEST:SIMPLE(filecheck=CHECK_SPV): -stage compute -entry computeMain -allow-glsl -target spirv -emit-spirv-directly

// This code should error since memory qualifiers are only allowed inside:
// Shader storage blocks, variables declared within shader storage blocks 
// and variables declared as image types. Named structs inside a Interface
// block violates these rules
// CHECK_GLSL: error
// CHECK_SPV: error
buffer Block4 {
    struct StructTmp
    {
        readonly int val;
    };
    readonly structTmp myStruct;
    coherent readonly vec4 member1;
    coherent vec4 member2;
}inputBuffer4;

layout(local_size_x = 1) in;
void computeMain()
{
}