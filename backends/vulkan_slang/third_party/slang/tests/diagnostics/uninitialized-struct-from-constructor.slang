//TEST:SIMPLE(filecheck=CHK): -target spirv -entry computeMain

struct TangentSpace
{
    static const float3 localNormal = {0, 1, 0};

    float4x4 tangentTransform;
    float3 geometryNormal;
	
    __init(in float3 normal, in float3 inRay)
    {
        // Should not warn here
        tangentTransform = getMatrix(normal, inRay);
        geometryNormal = localNormal;
    }

    float4x4 getMatrix(in float3 normal, in float3 inRay)
    {
        return float4x4(0.0f);
    }
}

//CHK-NOT: warning 41020
//CHK-NOT: warning 41021

[Shader("compute")]
[NumThreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
}

