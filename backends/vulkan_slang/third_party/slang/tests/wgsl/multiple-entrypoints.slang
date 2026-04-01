//TEST:SIMPLE(filecheck=WGSL): -target wgsl

//WGSL-DAG: @builtin(global_invocation_id)
//WGSL-DAG: @compute

[shader("compute")]
void main_compute(uint3 dtid : SV_DispatchThreadID) 
{
    // Empty compute shader
}

//WGSL-DAG: @builtin(front_facing)
//WGSL-DAG: @fragment

[shader("fragment")]
float4 main_fragment(bool isFront : SV_IsFrontFace) : SV_Target 
{
    return float4(1,1,1,1);
}

//WGSL-DAG: @builtin(vertex_index)
//WGSL-DAG: @vertex

[shader("vertex")]
float4 main_vertex(uint vertexID : SV_VertexID) : SV_Position 
{
    return float4(1,1,1,1);
}
