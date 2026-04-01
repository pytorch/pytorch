//TEST:SIMPLE(filecheck=CHECK): -target hlsl -stage raygeneration -entry main

RaytracingAccelerationStructure as;

RWStructuredBuffer<float> outputBuffer1;

// addComplexity ensures the code does not optimize any `someInData1`
void addComplexity1(inout float someInData1)
{
    someInData1 = outputBuffer1[1];
}
void addComplexity2(float dataIn)
{
    outputBuffer1[2] = dataIn;
}

void main()
{
    RayDesc ray;
    ray.Origin = float3(0);
    ray.TMin = 1e-6;
    ray.Direction = float3(0,0,1);
    ray.TMax = 100.0f;

    // CHECK: RayPayload_t{{_[0-9]}} rayPayload{{_[0-9]}};
    float someInData1 = 5.0f;
    addComplexity1(someInData1);

    // CHECK: rayPayload{{_[0-9]}}.data{{_[0-9]}} = {{.*}}
    // CHECK: TraceRay(
    // CHECK: {{.*}} = rayPayload{{.*}}.data{{.*}};
    TraceRay(as,
             1,
             0xff,
             0,
             0,
             2,
             ray,
             someInData1);
    outputBuffer1[0] = outputBuffer1[0]+someInData1;

    // CHECK: rayPayload{{_[0-9]}}.data{{_[0-9]}} = {{.*}}
    // CHECK: TraceMotionRay(
    // CHECK: {{.*}} = rayPayload{{.*}}.data{{.*}};
    TraceMotionRay(as,
            1,
            0xff,
            0,
            0,
            2,
            ray,
            0.0f,
            someInData1);
    outputBuffer1[0] = outputBuffer1[0]+someInData1;

    // CHECK: rayPayload{{_[0-9]}}.data{{_[0-9]}} = {{.*}}
    // CHECK: NvTraceRayHitObject(
    // CHECK: {{.*}} = rayPayload{{.*}}.data{{.*}};
    HitObject::TraceRay(as,
             1,
             0xff,
             0,
             0,
             2,
             ray,
             someInData1);
    outputBuffer1[0] = outputBuffer1[0]+someInData1;

    // CHECK: rayPayload{{_[0-9]}}.data{{_[0-9]}} = {{.*}}
    // CHECK: TraceMotionRay(
    // CHECK: {{.*}} = rayPayload{{.*}}.data{{.*}};
    HitObject::TraceMotionRay(as,
            1,
            0xff,
            0,
            0,
            2,
            ray,
            0.0f,
            someInData1);
    outputBuffer1[0] = outputBuffer1[0]+someInData1;

    // CHECK: rayPayload{{_[0-9]}}.data{{_[0-9]}} = {{.*}}
    // CHECK: NvInvokeHitObject(
    // CHECK: {{.*}} = rayPayload{{.*}}.data{{.*}};
    HitObject hitObject_HitOrMiss;
    HitObject::Invoke(
        as,
        hitObject_HitOrMiss,
        someInData1);
    outputBuffer1[0] = outputBuffer1[0]+someInData1;

    addComplexity2(someInData1);
}
