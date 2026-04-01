// sv-clip-distance.slang

//TEST:SIMPLE(filecheck=CHECK): -profile sm_5_0 -stage vertex -entry mainVertex -target glsl

//CHECK: out float  gl_ClipDistance[2];
//CHECK: gl_ClipDistance =

struct VertexInput 
{
    float3 position : POSITION; 
};

void mainVertex(VertexInput vi, out float4 outPosition : SV_Position, out float outClips[2] : SV_ClipDistance) 
{
    outPosition = float4(vi.position, 1);
    outClips[0] = vi.position.x / 10;
    outClips[1] = vi.position.z * 2;
} 

