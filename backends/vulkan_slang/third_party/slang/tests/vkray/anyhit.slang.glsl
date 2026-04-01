#version 460
#extension GL_EXT_ray_tracing : require
layout(row_major) uniform;
layout(row_major) buffer;
struct Params_0
{
    int mode_0;
};

layout(binding = 0)
layout(std140) uniform block_Params_0
{
    int mode_0;
}gParams_0;
layout(binding = 1)
uniform texture2D gParams_alphaMap_0;

layout(binding = 2)
uniform sampler gParams_sampler_0;

struct SphereHitAttributes_0
{
    vec3 normal_0;
};

hitAttributeEXT SphereHitAttributes_0 _S1;

struct ShadowRay_0
{
    vec4 hitDistance_0;
    vec3 dummyOut_0;
};

rayPayloadInEXT ShadowRay_0 _S2;

void main()
{
    if(gParams_0.mode_0 != 0)
    {
        if((textureLod(sampler2D(gParams_alphaMap_0,gParams_sampler_0), (_S1.normal_0.xy), (0.0)).x) > 0.0)
        {
            terminateRayEXT;;
        }
        else
        {
            ignoreIntersectionEXT;;
        }
    }

    vec3 _S3 = (gl_HitTriangleVertexPositionsEXT[(0U)]);
    _S2.dummyOut_0 = _S3;
    vec3 _S4 = (gl_HitTriangleVertexPositionsEXT[(1U)]);
    vec3 _S5 = _S3 + _S4;
    _S2.dummyOut_0 = _S5;
    vec3 _S6 = (gl_HitTriangleVertexPositionsEXT[(2U)]);
    _S2.dummyOut_0 = _S5 + _S6;
    return;
}

