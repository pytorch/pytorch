#pragma pack_matrix(column_major)
#ifdef SLANG_HLSL_ENABLE_NVAPI
#include "nvHLSLExtns.h"
#endif

#ifndef __DXC_VERSION_MAJOR
// warning X3557: loop doesn't seem to do anything, forcing loop to unroll
#pragma warning(disable : 3557)
#endif

struct SLANG_ParameterGroup_C_0
{
    float3 origin_0;
    float tMin_0;
    float3 direction_0;
    float tMax_0;
    uint rayFlags_0;
    uint instanceMask_0;
    uint shouldStopAtFirstHit_0;
};

cbuffer C_0 : register(b0)
{
    SLANG_ParameterGroup_C_0 C_0;
}
RaytracingAccelerationStructure myAccelerationStructure_0 : register(t0);

RWStructuredBuffer<int > resultBuffer_0 : register(u0);

struct MyRayPayload_0
{
    int value_0;
};

MyRayPayload_0 MyRayPayload_x24init_0(int value_1)
{
    MyRayPayload_0 _S1;
    _S1.value_0 = value_1;
    return _S1;
}

RayDesc RayDesc_x24init_0(float3 Origin_0, float TMin_0, float3 Direction_0, float TMax_0)
{
    RayDesc _S2;
    _S2.Origin = Origin_0;
    _S2.TMin = TMin_0;
    _S2.Direction = Direction_0;
    _S2.TMax = TMax_0;
    return _S2;
}

struct MyProceduralHitAttrs_0
{
    int value_2;
};

MyProceduralHitAttrs_0 MyProceduralHitAttrs_x24init_0(int value_3)
{
    MyProceduralHitAttrs_0 _S3;
    _S3.value_2 = value_3;
    return _S3;
}

bool myProceduralIntersection_0(inout float tHit_0, inout MyProceduralHitAttrs_0 hitAttrs_0)
{
    return true;
}

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
    payload_2.value_0 = int(1);
    return;
}

void myProceduralClosestHit_0(inout MyRayPayload_0 payload_3, MyProceduralHitAttrs_0 attrs_0)
{
    payload_3.value_0 = attrs_0.value_2;
    return;
}

void myMiss_0(inout MyRayPayload_0 payload_4)
{
    payload_4.value_0 = int(0);
    return;
}

[shader("compute")][numthreads(1, 1, 1)]
void main(uint3 tid_0 : SV_DispatchThreadID)
{
    uint index_0 = tid_0.x;
    MyRayPayload_0 payload_5 = MyRayPayload_x24init_0(int(-1));
    RayQuery<512U > query_0;
    query_0.TraceRayInline(myAccelerationStructure_0, C_0.rayFlags_0, C_0.instanceMask_0, RayDesc_x24init_0(C_0.origin_0, C_0.tMin_0, C_0.direction_0, C_0.tMax_0));
    MyProceduralHitAttrs_0 committedProceduralAttrs_0;
    MyProceduralHitAttrs_0 _S4 = MyProceduralHitAttrs_x24init_0(int(0));
    for(;;)
    {
        bool _S5 = query_0.Proceed();
        if(!_S5)
        {
            break;
        }
        uint _S6 = query_0.CandidateType();
        MyProceduralHitAttrs_0 committedProceduralAttrs_1;
        switch(_S6)
        {
        case 1U:
            {
                MyProceduralHitAttrs_0 candidateProceduralAttrs_0 = _S4;
                float tHit_1 = 0.0;
                bool _S7 = myProceduralIntersection_0(tHit_1, candidateProceduralAttrs_0);
                if(_S7)
                {
                    bool _S8 = myProceduralAnyHit_0(payload_5);
                    if(_S8)
                    {
                        query_0.CommitProceduralPrimitiveHit(tHit_1);
                        MyProceduralHitAttrs_0 _S9 = candidateProceduralAttrs_0;
                        if((C_0.shouldStopAtFirstHit_0) != 0U)
                        {
                            query_0.Abort();
                        }
                        committedProceduralAttrs_1 = _S9;
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
                bool _S10 = myTriangleAnyHit_0(payload_5);
                if(_S10)
                {
                    query_0.CommitNonOpaqueTriangleHit();
                    if((C_0.shouldStopAtFirstHit_0) != 0U)
                    {
                        query_0.Abort();
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
    uint _S11 = query_0.CommittedStatus();
    switch(_S11)
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
    resultBuffer_0[index_0] = payload_5.value_0;
    return;
}

