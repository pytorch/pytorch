//TEST:SIMPLE(filecheck=CHECK_GLSL): -allow-glsl -stage miss -entry main -target glsl
//TEST:SIMPLE(filecheck=CHECK_SPV): -allow-glsl -emit-spirv-directly -stage miss -entry main -target spirv-assembly

layout(binding = 0) uniform accelerationStructureEXT as;

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
buffer MyBlockName
{
    uint data[];
} outputBuffer;

// CHECK_GLSL-DAG: rayPayloadInEXT
// CHECK_SPV-DAG: IncomingRayPayload{{NV|KHR}}
layout(location = 2) rayPayloadInEXT vec4 payload;
// CHECK_GLSL-DAG: hitObjectAttributeNV
// CHECK_SPV-DAG: HitObjectAttributeNV
layout(location = 2) hitObjectAttributeNV vec4 attrMain;

bool testVars() {
    payload = vec4(1);
    attrMain = vec4(1);
    return true
// CHECK_GLSL-DAG: gl_LaunchIDNV
// CHECK_SPV-DAG-N: LaunchIdNV
        && gl_LaunchIDNV != uint3(99)
// CHECK_GLSL-DAG: gl_LaunchIDEXT
// spirv checked with CHECK_SPV-DAG-N
        && gl_LaunchIDEXT != uint3(99)
// CHECK_GLSL-DAG: gl_LaunchSizeNV
// CHECK_SPV-DAG-N: LaunchSizeNV
        && gl_LaunchSizeNV != uint3(99)
// CHECK_GLSL-DAG: gl_LaunchSizeEXT
// spirv checked with CHECK_SPV-DAG-N
        && gl_LaunchSizeEXT != uint3(99)
// CHECK_GLSL-DAG: gl_WorldRayOriginEXT
// CHECK_SPV-DAG: WorldRayOrigin{{NV|KHR}}
        && gl_WorldRayOriginEXT != vec3(99)
// CHECK_GLSL-DAG: gl_WorldRayDirectionEXT
// CHECK_SPV-DAG: WorldRayDirection{{NV|KHR}}
        && gl_WorldRayDirectionEXT != vec3(99)
// CHECK_GLSL-DAG: gl_RayTminEXT
// CHECK_SPV-DAG: RayTmin{{NV|KHR}}
        && gl_RayTminEXT != 99.0f
// CHECK_GLSL-DAG: gl_RayTmaxEXT
// CHECK_SPV-DAG: RayTmax{{NV|KHR}}
        && gl_RayTmaxEXT != 99.0f
// CHECK_GLSL-DAG: gl_IncomingRayFlagsEXT
// CHECK_SPV-DAG: IncomingRayFlags{{NV|KHR}}
        && gl_IncomingRayFlagsEXT != 0xFFFFFFFF
// CHECK_GLSL-DAG: gl_CurrentRayTimeNV
// CHECK_SPV-DAG: CurrentRayTimeNV
        && gl_CurrentRayTimeNV != 0.0f
        ;
}

void main()
{
    outputBuffer.data[0] = true
        && testVars()
        ;
    //BUF: 1
}
