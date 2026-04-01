#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : require
layout(row_major) uniform;
layout(row_major) buffer;
struct SLANG_ParameterGroup_C_0
{
    vec3 origin_0;
    float tMin_0;
    vec3 direction_0;
    float tMax_0;
    uint rayFlags_0;
    uint instanceMask_0;
    uint shouldStopAtFirstHit_0;
};

layout(binding = 2)
layout(std140) uniform _S1
{
    vec3 origin_0;
    float tMin_0;
    vec3 direction_0;
    float tMax_0;
    uint rayFlags_0;
    uint instanceMask_0;
    uint shouldStopAtFirstHit_0;
}C_0;
layout(binding = 0)
uniform accelerationStructureEXT myAccelerationStructure_0;

layout(std430, binding = 1) buffer StructuredBuffer_int_t_0 {
    int _data[];
} resultBuffer_0;
struct MyProceduralHitAttrs_0
{
    int value_0;
};

bool myProceduralIntersection_0(inout float tHit_0, inout MyProceduralHitAttrs_0 hitAttrs_0)
{
    return true;
}

struct MyRayPayload_0
{
    int value_1;
};

bool myProceduralAnyHit_0(inout MyRayPayload_0 payload_0)
{
    return true;
}

bool myTriangleAnyHit_0(inout MyRayPayload_0 payload_1)
{
    return true;
}

void myTriangleClosestHit_0(inout MyRayPayload_0 payload_2)
{
    payload_2.value_1 = 1;
    return;
}

void myProceduralClosestHit_0(inout MyRayPayload_0 payload_3, MyProceduralHitAttrs_0 attrs_0)
{
    payload_3.value_1 = attrs_0.value_0;
    return;
}

void myMiss_0(inout MyRayPayload_0 payload_4)
{
    payload_4.value_1 = 0;
    return;
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main()
{
    uint index_0 = gl_GlobalInvocationID.x;
    rayQueryEXT query_0;
    MyRayPayload_0 payload_5;
    payload_5.value_1 = -1;
    rayQueryInitializeEXT((query_0), (myAccelerationStructure_0), (C_0.rayFlags_0 | 512U), (C_0.instanceMask_0), (C_0.origin_0), (C_0.tMin_0), (C_0.direction_0), (C_0.tMax_0));
    MyProceduralHitAttrs_0 committedProceduralAttrs_0;
    for(;;)
    {
        bool _S2 = rayQueryProceedEXT(query_0);
        if(!_S2)
        {
            break;
        }
        uint _S3 = (rayQueryGetIntersectionTypeEXT((query_0), false));
        MyProceduralHitAttrs_0 committedProceduralAttrs_1;
        switch(_S3)
        {
        case 1U:
            {
                MyProceduralHitAttrs_0 candidateProceduralAttrs_0;
                candidateProceduralAttrs_0.value_0 = 0;
                float tHit_1 = 0.0;
                bool _S4 = myProceduralIntersection_0(tHit_1, candidateProceduralAttrs_0);
                if(_S4)
                {
                    bool _S5 = myProceduralAnyHit_0(payload_5);
                    if(_S5)
                    {
                        rayQueryGenerateIntersectionEXT(query_0, tHit_1);
                        MyProceduralHitAttrs_0 _S6 = candidateProceduralAttrs_0;
                        if(C_0.shouldStopAtFirstHit_0 != 0U)
                        {
                            rayQueryTerminateEXT(query_0);
                        }
                        committedProceduralAttrs_1 = _S6;
                    }
                    else
                    {
                        committedProceduralAttrs_1 = committedProceduralAttrs_0;
                    }
                }
                else
                {
                    committedProceduralAttrs_1 = committedProceduralAttrs_0;
                }
                break;
            }
        case 0U:
            {
                bool _S7 = myTriangleAnyHit_0(payload_5);
                if(_S7)
                {
                    rayQueryConfirmIntersectionEXT(query_0);
                    if(C_0.shouldStopAtFirstHit_0 != 0U)
                    {
                        rayQueryTerminateEXT(query_0);
                    }
                }
                committedProceduralAttrs_1 = committedProceduralAttrs_0;
                break;
            }
        default:
            {
                committedProceduralAttrs_1 = committedProceduralAttrs_0;
                break;
            }
        }
        committedProceduralAttrs_0 = committedProceduralAttrs_1;
    }
    uint _S8 = (rayQueryGetIntersectionTypeEXT((query_0), true));
    switch(_S8)
    {
    case 1U:
        {
            myTriangleClosestHit_0(payload_5);
            break;
        }
    case 2U:
        {
            myProceduralClosestHit_0(payload_5, committedProceduralAttrs_0);
            break;
        }
    case 0U:
        {
            myMiss_0(payload_5);
            break;
        }
    default:
        {
            break;
        }
    }
    resultBuffer_0._data[index_0] = payload_5.value_1;
    return;
}


