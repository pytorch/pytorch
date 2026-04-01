// sv-clip-distance-split.slang

//TEST:SIMPLE(filecheck=CHECK): -profile sm_5_0 -stage vertex -entry mainVertex -target glsl

//CHECK: out float  gl_ClipDistance[3];
//CHECK: gl_ClipDistance[0] =
//CHECK: gl_ClipDistance[1] =
//CHECK: gl_ClipDistance[2] =

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

VertexOutput mainVertex(VertexInput vi, out float outClip : SV_ClipDistance2) 
{ 
    VertexOutput vo;

    vo.position = float4(vi.position, 1);
    vo.clip0 = vi.position.x / 10;
    vo.clip1 = vi.position.z * 2;
    
    outClip = vi.position.y + 1;
    
    return vo; 
} 

