//TEST:SIMPLE(filecheck=CHECK_GLSL): -allow-glsl -stage raygeneration -entry main -target glsl
//TEST:SIMPLE(filecheck=CHECK_SPV): -allow-glsl -emit-spirv-directly -stage raygeneration -entry main -target spirv-assembly

layout(binding = 0) uniform accelerationStructureEXT as;

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
buffer MyBlockName
{
    uint data[];
} outputBuffer;

// CHECK_GLSL-DAG: rayPayloadEXT
// CHECK_SPV-DAG: RayPayload{{NV|KHR}}
layout(location = 2) rayPayloadEXT vec4 payload;
// CHECK_GLSL-DAG: hitObjectAttributeNV
// CHECK_SPV-DAG: HitObjectAttributeNV
layout(location = 2) hitObjectAttributeNV vec4 attrMain;
// CHECK_GLSL-DAG: callableDataEXT
// CHECK_SPV-DAG: CallableData{{NV|KHR}}
layout(location = 0) callableDataEXT vec4 outcall;

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
bool equals(matrix<T, N, M> lhs, matrix<T, N, M> rhs)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            if (
                lhs[i][j] 
                !=
                rhs[i][j]
                )
            {
                return false;
            }
        }
    }
    return true;
}

bool testHitObjectTraceRay() {
    hitObjectNV hit;
// CHECK_GLSL-DAG: hitObjectTraceRayNV
// CHECK_SPV-DAG: OpHitObjectTraceRayNV
    hitObjectTraceRayNV(hit, as, gl_RayFlagsNoneEXT, 0xff, 0, 0, 0, vec3(0.1, 0.1, 0.0), 0.01f, vec3(0, 0, 1), 1e4f, 2);
    return true
// CHECK_GLSL-DAG: hitObjectIsHitNV
// CHECK_SPV-DAG: OpHitObjectIsHitNV
        && hitObjectIsHitNV(hit) == true
        ;
}

bool testHitObjectTraceRayWithAuxInfo() {
    hitObjectNV hit;
// CHECK_GLSL-DAG: hitObjectTraceRayNV
// CHECK_SPV-DAG: OpHitObjectTraceRayNV
    hitObjectTraceRayNV(hit, as, gl_RayFlagsNoneEXT, 0xff, 0, 0, 0, vec3(0.1, 0.1, 0.0), 0.01f, vec3(0, 0, 1), 1e4f, 2);
// CHECK_GLSL-DAG: hitObjectGetAttributesNV
// CHECK_SPV-DAG: OpHitObjectGetAttributesNV
    hitObjectGetAttributesNV(hit, 2);
    vec4 testCopy = attrMain;

// CHECK_GLSL-DAG: reorderThreadNV
// CHECK_SPV-DAG: OpReorderThreadWithHintNV
    reorderThreadNV(0, 3);
// CHECK_GLSL-DAG: reorderThreadNV
// CHECK_SPV-DAG: OpReorderThreadWithHitObjectNV
    reorderThreadNV(hit);
// CHECK_GLSL-DAG: reorderThreadNV
// CHECK_SPV-DAG: OpReorderThreadWithHitObjectNV
    reorderThreadNV(hit, 0, 3);

    return true
// CHECK_GLSL-DAG: hitObjectGetShaderBindingTableRecordIndexNV
// CHECK_SPV-DAG: OpHitObjectGetShaderBindingTableRecordIndexNV
        && hitObjectGetShaderBindingTableRecordIndexNV(hit) == 0
// CHECK_GLSL-DAG: hitObjectGetShaderRecordBufferHandleNV
// CHECK_SPV-DAG: OpHitObjectGetShaderRecordBufferHandleNV
        && hitObjectGetShaderRecordBufferHandleNV(hit) != uvec2(0) // valid Handles are never 0, but not predictable either
        && testCopy == vec4(1)
// CHECK_GLSL-DAG: hitObjectIsHitNV
// CHECK_SPV-DAG: OpHitObjectIsHitNV
        && hitObjectIsHitNV(hit) == true
// CHECK_GLSL-DAG: hitObjectGetRayTMinNV
// CHECK_SPV-DAG: OpHitObjectGetRayTMinNV
        && hitObjectGetRayTMinNV(hit) == 0.01f
// CHECK_GLSL-DAG: hitObjectGetRayTMaxNV
// CHECK_SPV-DAG: OpHitObjectGetRayTMaxNV
        && hitObjectGetRayTMaxNV(hit) == 1e4f
// CHECK_GLSL-DAG: hitObjectGetObjectRayOriginNV
// CHECK_SPV-DAG: OpHitObjectGetObjectRayOriginNV
        && hitObjectGetObjectRayOriginNV(hit) == vec3(0.1, 0.1, 0.0) 
// CHECK_GLSL-DAG: hitObjectGetObjectRayDirectionNV
// CHECK_SPV-DAG: OpHitObjectGetObjectRayDirectionNV
        && hitObjectGetObjectRayDirectionNV(hit) == vec3(0, 0, 1)
// CHECK_GLSL-DAG: hitObjectGetWorldRayOriginNV
// CHECK_SPV-DAG: OpHitObjectGetWorldRayOriginNV
        && hitObjectGetWorldRayOriginNV(hit) == vec3(0.1, 0.1, 0.0) 
// CHECK_GLSL-DAG: hitObjectGetWorldRayDirectionNV
// CHECK_SPV-DAG: OpHitObjectGetWorldRayDirectionNV
        && hitObjectGetWorldRayDirectionNV(hit) == vec3(0, 0, 1)
// CHECK_GLSL-DAG: hitObjectGetObjectToWorldNV
// CHECK_SPV-DAG: OpHitObjectGetObjectToWorldNV
        && !equals(hitObjectGetObjectToWorldNV(hit), mat4x3(0))
// CHECK_GLSL-DAG: hitObjectGetWorldToObjectNV
// CHECK_SPV-DAG: OpHitObjectGetWorldToObjectNV
        && !equals(hitObjectGetWorldToObjectNV(hit), mat4x3(0))
// CHECK_GLSL-DAG: hitObjectGetInstanceCustomIndexNV
// CHECK_SPV-DAG: OpHitObjectGetInstanceCustomIndexNV
        && hitObjectGetInstanceCustomIndexNV(hit) == 0
// CHECK_GLSL-DAG: hitObjectGetInstanceIdNV
// CHECK_SPV-DAG: OpHitObjectGetInstanceIdNV
        && hitObjectGetInstanceIdNV(hit) == 0
// CHECK_GLSL-DAG: hitObjectGetGeometryIndexNV
// CHECK_SPV-DAG: OpHitObjectGetGeometryIndexNV
        && hitObjectGetGeometryIndexNV(hit) == 0
// CHECK_GLSL-DAG: hitObjectGetPrimitiveIndexNV
// CHECK_SPV-DAG: OpHitObjectGetPrimitiveIndexNV
        && hitObjectGetPrimitiveIndexNV(hit) == 0
// CHECK_GLSL-DAG: hitObjectGetHitKindNV
// CHECK_SPV-DAG: OpHitObjectGetHitKindNV
        && hitObjectGetHitKindNV(hit) == gl_HitKindFrontFacingTriangleEXT
        ;
}

bool testHitObjectTraceRayLaunchSubShaders() {
    hitObjectNV hit;
// CHECK_GLSL-DAG: hitObjectTraceRayNV
// CHECK_SPV-DAG: OpHitObjectTraceRayNV
    hitObjectTraceRayNV(hit, as, gl_RayFlagsNoneEXT, 0xff, 0, 0, 0, vec3(0.1, 0.1, 0.0), 0.01f, vec3(0, 0, 1), 1e4f, 2);
// CHECK_GLSL-DAG: hitObjectExecuteShaderNV
// CHECK_SPV-DAG: OpHitObjectExecuteShaderNV
    hitObjectExecuteShaderNV(hit, 2);
    return true
// CHECK_GLSL-DAG: hitObjectIsHitNV
// CHECK_SPV-DAG: OpHitObjectIsHitNV
        && hitObjectIsHitNV(hit) == true
        ;
}

bool testHitObjectTraceRayMotion() {
    hitObjectNV hit;
// CHECK_GLSL-DAG: hitObjectTraceRayMotionNV
// CHECK_SPV-DAG: OpHitObjectTraceRayMotionNV
    hitObjectTraceRayMotionNV(hit, as, gl_RayFlagsNoneEXT, 0xff, 0, 0, 0, vec3(0.1, 0.1, 0.0), 0.01f, vec3(0, 0, 1), 1e4f, 0.0f, 2);
    return true
// CHECK_GLSL-DAG: hitObjectIsHitNV
// CHECK_SPV-DAG: OpHitObjectIsHitNV
        && hitObjectIsHitNV(hit) == true
// CHECK_GLSL-DAG: hitObjectGetCurrentTimeNV
// CHECK_SPV-DAG: OpHitObjectGetCurrentTimeNV
        && hitObjectGetCurrentTimeNV(hit) == 0.0f
        ;
}

bool testTraceRayFunctions() {
// CHECK_GLSL-DAG: traceRayEXT
// CHECK_SPV-DAG: OpTraceRayKHR
    traceRayEXT(as, 0, 0, 0, 0, 0, vec3(0), 0, vec3(0), 0, 2);
    return true
        ;
}

bool testTraceRayCalls() {
    return true
        && testHitObjectTraceRay()
        && testHitObjectTraceRayWithAuxInfo()
        && testHitObjectTraceRayLaunchSubShaders()
        && testHitObjectTraceRayMotion()
        && testTraceRayFunctions()
        ;
}

bool testHitObjectRecord() {
    hitObjectNV hit;
// CHECK_GLSL-DAG: hitObjectRecordHitNV
// CHECK_SPV-DAG: OpHitObjectRecordHitNV
    hitObjectRecordHitNV(hit, as, 0, 0, 0, gl_HitKindFrontFacingTriangleEXT, 0, 0, vec3(0.1, 0.1, 0.0), 0.01f, vec3(0, 0, 1), 1e4f, 2);
    return true
// CHECK_GLSL-DAG: hitObjectIsHitNV
// CHECK_SPV-DAG: OpHitObjectIsHitNV
        && hitObjectIsHitNV(hit) == true
        ;
}

bool testHitObjectRecordIndex() {
    hitObjectNV hit;
// CHECK_GLSL-DAG: hitObjectRecordHitWithIndexNV
// CHECK_SPV-DAG: OpHitObjectRecordHitWithIndexNV
    hitObjectRecordHitWithIndexNV(hit, as, 0, 0, 0, gl_HitKindFrontFacingTriangleEXT, 0, vec3(0.1, 0.1, 0.0), 0.01f, vec3(0, 0, 1), 1e4f, 2);
    return true
// CHECK_GLSL-DAG: hitObjectIsHitNV
// CHECK_SPV-DAG: OpHitObjectIsHitNV
        && hitObjectIsHitNV(hit) == true
        ;
}

bool testHitObjectRecordMotion() {
    hitObjectNV hit;
// CHECK_GLSL-DAG: hitObjectRecordHitMotionNV
// CHECK_SPV-DAG: OpHitObjectRecordHitMotionNV
    hitObjectRecordHitMotionNV(hit, as, 0, 0, 0, gl_HitKindFrontFacingTriangleEXT, 0, 0, vec3(0.1, 0.1, 0.0), 0.01f, vec3(0, 0, 1), 1e4f, 0.0f, 2);
    return true
// CHECK_GLSL-DAG: hitObjectIsHitNV
// CHECK_SPV-DAG: OpHitObjectIsHitNV
        && hitObjectIsHitNV(hit) == true
        ;
}

bool testHitObjectRecordIndexMotion() {
    hitObjectNV hit;
// CHECK_GLSL-DAG: hitObjectRecordHitWithIndexMotionNV
// CHECK_SPV-DAG: OpHitObjectRecordHitWithIndexMotionNV
    hitObjectRecordHitWithIndexMotionNV(hit, as, 0, 0, 0, gl_HitKindFrontFacingTriangleEXT, 0, vec3(0.1, 0.1, 0.0), 0.01f, vec3(0, 0, 1), 1e4f, 0.0f, 2);
    return true
// CHECK_GLSL-DAG: hitObjectIsHitNV
// CHECK_SPV-DAG: OpHitObjectIsHitNV
        && hitObjectIsHitNV(hit) == true
        ;
}

bool testRecordHitCalls() {
    return true
    && testHitObjectRecord()
    && testHitObjectRecordIndex()
    && testHitObjectRecordMotion()
    && testHitObjectRecordIndexMotion()
    ;
}

bool testHitObjectRecordMiss() {
    hitObjectNV hit;
// CHECK_GLSL-DAG: hitObjectRecordMissNV
// CHECK_SPV-DAG: OpHitObjectRecordMissNV
    hitObjectRecordMissNV(hit, 0, vec3(0.1, 0.1, 0.0), 0.01f, vec3(0, 0, 1), 1e4f);
    return true
// CHECK_GLSL-DAG: hitObjectIsMissNV
// CHECK_SPV-DAG: OpHitObjectIsMissNV
        && hitObjectIsMissNV(hit) == true
        ;
}

bool testHitObjectRecordMissMotion() {
    hitObjectNV hit;
// CHECK_GLSL-DAG: hitObjectRecordMissMotionNV
// CHECK_SPV-DAG: OpHitObjectRecordMissMotionNV
    hitObjectRecordMissMotionNV(hit, 0, vec3(0.1, 0.1, 0.0), 0.01f, vec3(0, 0, 1), 1e4f, 0.0f);
    return true
// CHECK_GLSL-DAG: hitObjectIsMissNV
// CHECK_SPV-DAG: OpHitObjectIsMissNV
        && hitObjectIsMissNV(hit) == true
        ;
}

bool testRecordMissCalls() {
    return true
    && testHitObjectRecordMiss()
    && testHitObjectRecordMissMotion()
    ;
}

bool testHitObjectRecordEmpty() {
    hitObjectNV hit;
// CHECK_GLSL-DAG: hitObjectRecordEmptyNV
// CHECK_SPV-DAG: OpHitObjectRecordEmptyNV
    hitObjectRecordEmptyNV(hit);
    return true
// CHECK_GLSL-DAG: hitObjectIsEmptyNV
// CHECK_SPV-DAG: OpHitObjectIsEmptyNV
        && hitObjectIsEmptyNV(hit) == true
        ;
}

bool testRecordEmptyCalls() {
    return true
    && testHitObjectRecordEmpty()
    ;
}

bool testTraceRayMotion() {
// CHECK_GLSL-DAG: traceRayMotionNV
// CHECK_SPV-DAG: OpTraceRayMotionNV
    traceRayMotionNV(as, gl_RayFlagsNoneEXT, 0xff, 0, 0, 0, vec3(0.1, 0.1, 0.0), 0.01f, vec3(0, 0, 1), 1e4f, 0.0f, 2);
    attrMain = vec4(1); //ensure traceRayMotionNV does not get optimized out 
    return true &&
        attrMain == vec4(1)
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
        && testTraceRayCalls()
        && testRecordHitCalls()
        && testRecordMissCalls()
        && testRecordEmptyCalls()
        && testTraceRayMotion()
        && testReadWriteOfObjects()
        ;    
    //BUF: 1
}
