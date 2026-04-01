//TEST:SIMPLE(filecheck=CHECK_GLSL): -allow-glsl -stage anyhit -entry main -target glsl
//TEST:SIMPLE(filecheck=CHECK_SPV): -allow-glsl -emit-spirv-directly -stage anyhit -entry main -target spirv-assembly

layout(binding = 0) uniform accelerationStructureEXT as;

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
buffer MyBlockName
{
    uint data[];
} outputBuffer;

bool testVars() {
    return true
// CHECK_GLSL-DAG: gl_HitTEXT
// CHECK_SPV-DAG: RayTmax{{NV|KHR}}
        && gl_HitTEXT != 99.0f
// CHECK_GLSL-DAG: gl_HitKindEXT
// CHECK_SPV-DAG: HitKind{{NV|KHR}}
        && gl_HitKindEXT != 0xFFFFFFFF
        ;    
}

void main()
{
    outputBuffer.data[0] = true
        && testVars()
        ;
    //BUF: 1
}
