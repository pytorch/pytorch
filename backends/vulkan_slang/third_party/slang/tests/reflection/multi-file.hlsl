//TEST:REFLECTION:-D__SLANG__ -entry mainVS -profile vs_4_0 -target hlsl tests/reflection/multi-file-extra.hlsl -entry mainFS -profile ps_4_0 -no-codegen

// Here we are testing the case where multiple translation units are provided
// at once, so that we want combined reflection information for the resulting
// program. The other part of this program is in `multi-file-extra.hlsl`.

#include "multi-file-defines.h"

#ifdef __SLANG__
import multi_file_shared;
#else
#include "multi-file-shared.slang"
#endif

Texture2D vertexT R(: register(t0));
SamplerState vertexS R(: register(s0));

BEGIN_CBUFFER(vertexC)
{
    float3 vertexCA;
    float  vertexCB;
    float3 vertexCC;
    float2 vertexCD;
}
END_CBUFFER(vertexC, register(b0))

float4 mainVS() : SV_POSITION
{
    // Go ahead and use everything here, just to make sure things got placed correctly
    return use(sharedT, sharedS)
        +  use(CBUFFER_REF(sharedC, sharedCD))
        +  use(vertexT, vertexS)
        +  use(CBUFFER_REF(vertexC, vertexCD))
        +  use(sharedTV, vertexS)
        ;
}
