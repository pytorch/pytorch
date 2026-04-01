//TEST_IGNORE_FILE:
#version 460
#extension GL_NV_ray_tracing : require
layout(row_major) uniform;
layout(row_major) buffer;
struct Sphere_0
{
    vec3 position_0;
    float radius_0;
};

struct SLANG_ParameterGroup_U_0
{
    Sphere_0 gSphere_0;
};

layout(binding = 0)
layout(std140) uniform _S1
{
    Sphere_0 gSphere_0;
} U_0;
struct RayDesc_0
{
    vec3 Origin_0;
    float TMin_0;
    vec3 Direction_0;
    float TMax_0;
};

struct SphereHitAttributes_0
{
    vec3 normal_0;
};

bool rayIntersectsSphere_0(RayDesc_0 ray_0, Sphere_0 sphere_0, out float tHit_0, out SphereHitAttributes_0 attrs_0)
{
    tHit_0 = sphere_0.radius_0;
    attrs_0.normal_0 = sphere_0.position_0;
    return tHit_0 >= ray_0.TMin_0;
}

hitAttributeNV
SphereHitAttributes_0 a_0;

bool ReportHit_0(float tHit_1, uint hitKind_0, SphereHitAttributes_0 attributes_0)
{
    a_0 = attributes_0;
    bool _S2 = reportIntersectionNV(tHit_1, hitKind_0);
    return _S2;
}

void main()
{
    RayDesc_0 ray_1;
    vec3 _S3 = ((gl_ObjectRayOriginNV));
    ray_1.Origin_0 = _S3;
    vec3 _S4 = ((gl_ObjectRayDirectionNV));
    ray_1.Direction_0 = _S4;
    float _S5 = ((gl_RayTminNV));
    ray_1.TMin_0 = _S5;
    float _S6 = ((gl_RayTmaxNV));
    ray_1.TMax_0 = _S6;
    float tHit_2;
    SphereHitAttributes_0 attrs_1;
    bool _S7 = rayIntersectsSphere_0(ray_1, U_0.gSphere_0, tHit_2, attrs_1);
    if(_S7)
    {
        bool _S8 = ReportHit_0(tHit_2, uint(0), attrs_1);
    }
    else
    {
    }
    return;
}
