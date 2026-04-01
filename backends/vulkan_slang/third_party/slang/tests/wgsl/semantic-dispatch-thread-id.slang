//TEST:SIMPLE(filecheck=WGSL): -target wgsl -stage compute -entry main

//WGSL-DAG: @builtin(global_invocation_id)
//WGSL-DAG: @compute

void main(uint3 dtid : SV_DispatchThreadID) 
{
    // Empty compute shader
}
