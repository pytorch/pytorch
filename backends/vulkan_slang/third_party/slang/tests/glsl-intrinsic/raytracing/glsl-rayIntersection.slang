//TEST:SIMPLE(filecheck=CHECK_GLSL): -stage intersection -entry main -target glsl -allow-glsl 
//TEST:SIMPLE(filecheck=CHECK_SPV): -stage intersection -entry main -target spirv-assembly -emit-spirv-directly -allow-glsl 

layout(binding = 0) uniform accelerationStructureEXT as;

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
buffer MyBlockName
{ 
    uint data[];
} outputBuffer;

//CHECK_GLSL-DAG: hitAttributeEXT
//CHECK_SPV-DAG: HitAttribute{{NV|KHR}}
hitAttributeEXT vec4 hitAttr;

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

bool testVars() {
    hitAttr = vec4(1);
    
    return true
// CHECK_GLSL-DAG: gl_PrimitiveID
// CHECK_SPV-DAG: PrimitiveId
        && gl_PrimitiveID != 10000
// CHECK_GLSL-DAG: gl_InstanceID
// CHECK_SPV-DAG: InstanceId
        && gl_InstanceID != 10000
// CHECK_GLSL-DAG: gl_InstanceCustomIndexEXT
// CHECK_SPV-DAG: InstanceCustomIndex{{NV|KHR}}
        && gl_InstanceCustomIndexEXT != 10000
// CHECK_GLSL-DAG: gl_GeometryIndexEXT
// CHECK_SPV-DAG: RayGeometryIndexKHR
        && gl_GeometryIndexEXT != 10000
// CHECK_GLSL-DAG: gl_ObjectRayOriginEXT
// CHECK_SPV-DAG: ObjectRayOrigin{{NV|KHR}}
        && gl_ObjectRayOriginEXT != vec3(99)
// CHECK_GLSL-DAG: gl_ObjectRayDirectionEXT
// CHECK_SPV-DAG: ObjectRayDirection{{NV|KHR}}
        && gl_ObjectRayDirectionEXT != vec3(99)
// CHECK_GLSL-DAG: gl_ObjectToWorldEXT
//COM: CHECK_SPV-DAG: ObjectToWorldNV
        && !equals(gl_ObjectToWorldEXT, mat4x3(0.0f))
// CHECK_GLSL-DAG: gl_ObjectToWorld3x4EXT
//COM: CHECK_SPV-DAG: ObjectToWorldNV
        && !equals(gl_ObjectToWorld3x4EXT, mat3x4(0.0f))
// CHECK_GLSL-DAG: gl_WorldToObjectEXT
// CHECK_SPV-DAG-N: WorldToObjectNV 
        && !equals(gl_WorldToObjectEXT, mat4x3(0.0f))
// CHECK_GLSL-DAG: gl_WorldToObject3x4EXT
// spirv checked with CHECK_SPV-DAG-N
        && !equals(gl_WorldToObject3x4EXT, mat3x4(0.0f))
        ;
}

bool testFunctions() {
    return true
// CHECK_GLSL-DAG: reportIntersectionEXT
// CHECK_SPV-DAG: OpReportIntersectionKHR
        && reportIntersectionEXT(99, 0);
}

void main()
{
    outputBuffer.data[0] = true
        && testVars()
        && testFunctions();
        ;
    //BUF: 1
}
