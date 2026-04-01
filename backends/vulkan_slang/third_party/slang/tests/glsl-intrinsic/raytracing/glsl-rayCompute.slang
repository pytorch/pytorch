//TEST:SIMPLE(filecheck=CHECK_GLSL): -allow-glsl -stage compute -entry computeMain -target glsl
//TEST:SIMPLE(filecheck=CHECK_SPV): -allow-glsl -emit-spirv-directly -stage compute -entry computeMain -target spirv-assembly

//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF): -allow-glsl -vk -compute -output-using-type -render-feature ray-query -entry computeMain 
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF): -allow-glsl -vk -compute -output-using-type -render-feature ray-query -emit-spirv-directly -entry computeMain 

// note: the referenced input data is in render-test-main kVertexData; 1 triangle at {(0,0,0.5),(0,1,0.5),(1,0,0.5)}
//TEST_INPUT: set scene = AccelerationStructure
layout(binding = 0) uniform accelerationStructureEXT scene;

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
buffer MyBlockName
{
    uint data[];
} outputBuffer;

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

bool testAbortRayTrace() {
    rayQueryEXT q;
    rayQueryInitializeEXT(q, scene, gl_RayFlagsNoneEXT, 0xff, vec3(0.1, 0.1, 0.0), 0.01f, vec3(0, 0, 1), 1e4f);
    rayQueryTerminateEXT(q);

    return true
        && rayQueryGetIntersectionTypeEXT(q, true) == gl_RayQueryCommittedIntersectionNoneEXT
        ;
}

bool testProceedRayTrace() {
    rayQueryEXT q;
    rayQueryInitializeEXT(q, scene, gl_RayFlagsNoneEXT, 0xff, vec3(0.1, 0.1, 0.0), 0.01f, vec3(0, 0, 1), 1e4f);
    rayQueryProceedEXT(q);

    return true
        && rayQueryGetIntersectionTypeEXT(q, true) == gl_RayQueryCommittedIntersectionTriangleEXT
        && rayQueryGetIntersectionTEXT(q, true) == 0.5
        ;
}

bool testProceedRayAuxInfo() {
    rayQueryEXT q;
    rayQueryInitializeEXT(q, scene, gl_RayFlagsNoneEXT, 0xff, vec3(0.1, 0.1, 0.0), 0.01f, vec3(0, 0, 1), 1e4f);
    rayQueryProceedEXT(q);

    return true
        && rayQueryGetRayTMinEXT(q) == 0.01f
        && rayQueryGetRayFlagsEXT(q) == gl_RayFlagsNoneEXT
        && rayQueryGetWorldRayOriginEXT(q) == vec3(0.1, 0.1, 0.0)
        && rayQueryGetWorldRayDirectionEXT(q) == vec3(0, 0, 1)
        && rayQueryGetIntersectionInstanceCustomIndexEXT(q, true) == 0
        && rayQueryGetIntersectionInstanceIdEXT(q, true) == 0
        && rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(q, true) == 0
        && rayQueryGetIntersectionGeometryIndexEXT(q, true) == 0
        && rayQueryGetIntersectionPrimitiveIndexEXT(q, true) == 0
        && rayQueryGetIntersectionBarycentricsEXT(q, true) == vec2(0.1)
        && rayQueryGetIntersectionFrontFaceEXT(q, true) == true
        && rayQueryGetIntersectionCandidateAABBOpaqueEXT(q) == true
        && rayQueryGetIntersectionObjectRayDirectionEXT(q, true) == vec3(0, 0, 1)
        && rayQueryGetIntersectionObjectRayOriginEXT(q, true) == vec3(0.1, 0.1, 0.0)
        && !equals(rayQueryGetIntersectionObjectToWorldEXT(q, true), mat4x3(0)) // some compile error to sort later
        && !equals(rayQueryGetIntersectionWorldToObjectEXT(q, true), mat4x3(0))
        ;
}
bool testRayTraceGenerate() {
    rayQueryEXT q;
    rayQueryInitializeEXT(q, scene, gl_RayFlagsNoneEXT, 0xff, vec3(0.1, 0.1, 0.0), 0.01f, vec3(0, 0, 1), 1e4f);
    rayQueryGenerateIntersectionEXT(q, 0.5);
    rayQueryConfirmIntersectionEXT(q);

    return true
        && rayQueryGetIntersectionTypeEXT(q, true) == gl_RayQueryCommittedIntersectionTriangleEXT
        ;
}

// CHECK_GLSL-DAG: main
// CHECK_SPV-DAG: OpEntryPoint
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void computeMain()
{
    outputBuffer.data[0] = true
        && testAbortRayTrace()
        //&& testProceedRayTrace()
        //&& testProceedRayAuxInfo()
        //&& testRayTraceGenerate()
        ;
    // BUF: 1
}

