//TEST:SIMPLE(filecheck=WGSL): -target wgsl -stage compute -entry main

//WGSL-DAG: @builtin(local_invocation_index)
//WGSL-DAG: @compute

void main(uint idx : SV_GroupIndex) 
{
    // Empty compute shader
}
