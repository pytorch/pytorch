//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -stage raygeneration -entry main -enable-experimental-passes

// Check to ensure we make global ray-tracing objects. Ensure we store into these variables directly and not through a context-pointer.

// CHECK-DAG: %[[RAYTRACING_AS_TYPE:[A-Za-z0-9_]+]] = OpTypeAccelerationStructureKHR
// CHECK-DAG: %[[RAYTRACING_AS_PTR_TYPE:[A-Za-z0-9_]+]] = OpTypePointer UniformConstant %[[RAYTRACING_AS_TYPE]]
// CHECK-DAG: %[[RAYTRACING_AS:[A-Za-z0-9_]+]] = OpVariable %[[RAYTRACING_AS_PTR_TYPE]] UniformConstant

// CHECK-DAG: %[[RAY_PAYLOAD:[A-Za-z0-9_]+]] = OpVariable %{{.*}} RayPayloadKHR

// CHECK: OpLoad {{.*}} %[[RAYTRACING_AS]]
// CHECK: OpTraceRayKHR{{.*}} %[[RAY_PAYLOAD]]

RaytracingAccelerationStructure as;

struct ShadowRay
{
    float hitDistance;
};

void nestedNestedCall()
{
    RayDesc ray = {};

    ShadowRay shadowRay;
    shadowRay.hitDistance = 0;

    TraceRay(as,
            // ray flags
            1,
            // cull mask
            0xff,
            // sbt record offset
            0,
            // sbt record stride
            0,
            // missIndex
            2,
            // ray
            ray,
            // payload
            shadowRay);
}

void nestedCall()
{
    nestedNestedCall();
}

void main()
{
    nestedCall();
}