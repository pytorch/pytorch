//TEST:SIMPLE(filecheck=WGSL): -target wgsl -stage vertex -entry main

//WGSL-DAG: @builtin(vertex_index)
//WGSL-DAG: @vertex

float4 main(uint vertexID : SV_VertexID) : SV_Position 
{
    return float4(1,1,1,1);
}
