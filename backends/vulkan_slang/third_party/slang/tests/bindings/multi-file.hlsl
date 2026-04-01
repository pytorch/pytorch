//TEST:COMPARE_HLSL: -profile sm_4_0 -entry main1 -stage vertex Tests/bindings/multi-file-extra.hlsl -entry main -stage fragment

// Here we are going to test that we can correctly generating bindings when we
// are presented with a program spanning multiple input files (and multiple entry points)

// This file provides the vertex shader, while the fragment shader resides in
// the file `multi-file-extra.hlsl`

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

float4 main1() : SV_POSITION
{
    // Go ahead and use everything here, just to make sure things got placed correctly
    return use(sharedT, sharedS)
        +  use(CBUFFER_REF(sharedC, sharedCD))
        +  use(vertexT, vertexS)
        +  use(CBUFFER_REF(vertexC, vertexCD))
        +  use(sharedTV, vertexS)
        ;
}