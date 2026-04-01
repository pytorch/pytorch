// sv-clip-distance.slang

//TEST:SIMPLE(filecheck=CHECK): -profile sm_5_0 -stage vertex -entry mainVertex -target glsl

//CHECK: out float  gl_ClipDistance[2];
//CHECK: gl_ClipDistance[0] =
//CHECK: gl_ClipDistance[1] =

struct VertexInput 
{
    float3 position : POSITION; 
};

struct VertexOutput
{ 
    float4 position : SV_Position; 
    
    float clip0 : SV_ClipDistance0;
    float clip1 : SV_ClipDistance1; 
}; 

VertexOutput mainVertex(VertexInput vi) 
{ 
    VertexOutput vo;

    vo.position = float4(vi.position, 1);
    vo.clip0 = vi.position.x / 10;
    vo.clip1 = vi.position.z * 2;
    
    return vo; 
} 

