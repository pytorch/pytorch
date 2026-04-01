//TEST:SIMPLE(filecheck=CHECK_GLSL): -allow-glsl -stage raygeneration -entry main -target glsl
//TEST:SIMPLE(filecheck=CHECK_SPV): -allow-glsl -emit-spirv-directly -stage raygeneration -entry main -target spirv-assembly

// CHECK_GLSL-DAG: hitObjectAttributeNV
// CHECK_SPV-DAG: HitObjectAttributeNV

layout(binding = 0) uniform accelerationStructureEXT as;

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
buffer MyBlockName
{
    uint data[];
} outputBuffer;

layout(location = 2) hitObjectAttributeNV vec4 attrMain;

bool testHitObjectRecord() {
    hitObjectNV hit;
    // CHECK_GLSL: hitObjectRecordHitNV
    // CHECK_SPV: OpHitObjectRecordHitNV
    hitObjectRecordHitNV(hit, as, 0, 0, 0, gl_HitKindFrontFacingTriangleEXT, 0, 0, vec3(0.1, 0.1, 0.0), 0.01f, vec3(0, 0, 1), 1e4f, 2);
    return true
           // CHECK_GLSL: hitObjectIsHitNV
           // CHECK_SPV: OpHitObjectIsHitNV
           && hitObjectIsHitNV(hit) == true
        ;
}

void main()
{
    outputBuffer.data[0] = true
        && testHitObjectRecord();
    //BUF: 1
}
