//TEST:SIMPLE(filecheck=CHECK): -allow-glsl -target spirv -emit-spirv-directly -stage raygeneration -entry main -enable-experimental-passes

// Check to ensure we make global ray-tracing objects. Ensure we store into these variables directly and not through a context-pointer.

// CHECK-DAG: %[[RAYTRACING_AS_TYPE:[A-Za-z0-9_]+]] = OpTypeAccelerationStructureKHR
// CHECK-DAG: %[[RAYTRACING_AS_PTR_TYPE:[A-Za-z0-9_]+]] = OpTypePointer UniformConstant %[[RAYTRACING_AS_TYPE]]
// CHECK-DAG: %[[RAYTRACING_AS:[A-Za-z0-9_]+]] = OpVariable %[[RAYTRACING_AS_PTR_TYPE]] UniformConstant

// CHECK-DAG: %[[RAY_PAYLOAD:[A-Za-z0-9_]+]] = OpVariable %{{.*}} RayPayloadKHR
// CHECK-DAG: %[[HIT_ATTR:[A-Za-z0-9_]+]] = OpVariable %{{.*}} HitObjectAttributeNV
// CHECK-DAG: %[[CALL_DATA:[A-Za-z0-9_]+]] = OpVariable %{{.*}} CallableDataKHR


// CHECK: OpLoad {{.*}} %[[RAYTRACING_AS]]
// CHECK: OpStore %[[RAY_PAYLOAD]]
// CHECK: OpStore %[[HIT_ATTR]]
// CHECK: OpStore %[[CALL_DATA]]

layout(binding = 0) uniform accelerationStructureEXT as;

buffer MyBlockName
{
    uint data[];
} outputBuffer;
layout(location = 2) rayPayloadEXT vec4 payload;
layout(location = 2) hitObjectAttributeNV vec4 attrMain;
layout(location = 0) callableDataEXT vec4 outcall;

bool testHitObjectTraceRay() {
    hitObjectNV hit;


    hitObjectTraceRayNV(hit, as, gl_RayFlagsNoneEXT, 0xff, 0, 0, 0, vec3(0.1, 0.1, 0.0), 0.01f, vec3(0, 0, 1), 1e4f, 2);
    return true
        && hitObjectIsHitNV(hit) == true
        ;
}


bool testPayloadReadWrite() {
    payload = vec4(2);
    vec4 read = payload;
    return true
        && read != vec4(0)
        ;
}

bool testAttributeReadWrite() {
    attrMain = vec4(2);
    vec4 read = attrMain;
    return true
        && read != vec4(0)
        ;
}

bool testCallableReadWrite() {
    outcall = vec4(2);
    vec4 read = outcall;
    return true
        && read != vec4(0)
        ;
}

bool testReadWriteOfObjects(){
    return true
        && testPayloadReadWrite()
        && testAttributeReadWrite()
        && testCallableReadWrite();
        ;       
}

void main()
{
    outputBuffer.data[0] = true
        && testHitObjectTraceRay()
        && testReadWriteOfObjects()
        ;
}
